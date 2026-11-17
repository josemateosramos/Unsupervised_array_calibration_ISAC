# -*- coding: utf-8 -*-
'''
Library of the unsupervised end-to-end ISAC calibration framework.

Notation follows the paper: K antennas, S subcarriers, gamma_x/p_x the true
GPIs/antenna positions (Xi), xi_x/omega_x their learnable counterparts (Psi),
f_s/f_c/f_tx the sensing/communication/ISAC precoders, Y_s/y_c the sensing and
communication observations, kappa the CSI at the UE. The suffixes _s and _c
denote sensing and communication quantities (the paper's tilde for the UE).

Sections:
    * Parameter objects : dataclasses bundling the simulation parameters.
    * Generic functions : random draws, steering/delay vectors and precoders.
    * Sensing functions : sensing channel and OMP receiver.
    * Set metric        : GOSPA (evaluation only).
    * Communication     : constellation, channel and ML decoder.
    * Training functions: projection onto the feasible set and training loop.
    * Testing functions : sensing, communication and ISAC evaluation.
    * Experiment helpers: file names, evaluation phases and saved results of methods/.
'''
import torch
import torch.nn.functional as F
import numpy as np
import itertools
import math
from dataclasses import dataclass, replace
from typing import Any

#######################################################################
# Parameter objects                                                   #
#######################################################################
#Use dataclasses.replace (re-exported by this module) to derive a modified copy,
#e.g. replace(system, P=P_new) for a transmit-power sweep.

@dataclass
class SectorParams:
    '''Distribution of the angle-range uncertainty regions. Per sample,
    theta_mean ~ U[theta_mean_min, theta_mean_max], Delta_theta ~ U[delta_theta_min,
    delta_theta_max] and [theta_min, theta_max] = theta_mean -+ Delta_theta/2
    (Table II). The range interval [R_min, R_max] is drawn in the same way.'''
    theta_mean_min: Any
    theta_mean_max: Any
    delta_theta_min: Any
    delta_theta_max: Any
    R_mean_min: Any
    R_mean_max: Any
    delta_R_min: Any
    delta_R_max: Any

@dataclass
class SystemParams:
    '''Physical and system constants (Table II) and true impairments Xi.'''
    T_max: Any          #Maximum number of targets
    T_max_c: Any        #Maximum number of BS-UE paths, including LoS
    P: Any              #Transmit power [W]
    sigma_mean: Any     #Mean RCS [m^2]
    R_cp: Any           #Range equivalent to the cyclic prefix, c*T_cp [m]
    K: Any              #Antennas per ULA
    S: Any              #Subcarriers
    delta_f: Any        #Subcarrier spacing [Hz]
    lamb: Any           #Wavelength [m]
    N_theta: Any        #Points of the angular grids (precoder and OMP)
    N_tau: Any          #Points of the OMP delay grid
    theta_grid: Any     #Angular grid over the field of view [-pi/2, pi/2]
    x_ref: Any          #All possible transmitted symbols
    noise_var: Any      #Noise variance N0*S*delta_f [W]
    gamma_tx: Any       #True TX GPIs. Shape: (K,1)
    gamma_rx: Any       #True RX GPIs. Shape: (K,1)
    p_tx: Any           #True TX antenna positions. Shape: (K,)
    p_rx: Any           #True RX antenna positions. Shape: (K,)

@dataclass
class TestParams:
    '''Sensing evaluation parameters (Sec. IV-B).'''
    p_fa_target: Any    #Target false-alarm probability
    delta_p_fa: Any     #Allowed deviation from p_fa_target
    delta_init: Any     #Initial OMP termination thresholds of the fixed-p_fa search
    mu_gospa: Any
    p_gospa: Any
    gamma_gospa: Any    #GOSPA cut-off distance

#######################################################################
# Generic functions                                                   #
#######################################################################

#Upper end of the baseline ROC thresholds as a function of the transmit power P (T_max = 5)
_ROC_UPPER = [(1.0, 4.7e-2), (0.1, 3e-3), (0.01, 1.1e-3), (1e-4, 8e-4)]

def getThresholdsROC(num_thr, P, device='cuda'):
    '''Log-spaced OMP termination thresholds delta to sweep the baseline ROC.'''
    upper = None
    for P_val, ub in _ROC_UPPER:
        if upper is None and torch.isclose(P, torch.tensor(P_val)):
            upper = ub
    if upper is None:
        raise ValueError(f'No ROC thresholds defined for P = {P.item()} W; add them to _ROC_UPPER')
    return torch.logspace(np.log10(2.50e-4), np.log10(upper), num_thr, device=device)

def generatePositionImpairments(eps_max, K, lamb, p_first, rng_gen=None, device='cuda'):
    '''Impaired antenna positions p_x = p_ideal + eps_p, with [eps_p]_k ~ U[-eps_max, eps_max]
    and p_ideal a lambda/2-spaced ULA starting at p_first. Shape: (K,)'''
    p_ideal = p_first + lamb/2.0 * torch.arange(0,K, device=device)
    eps_p = eps_max*(2*torch.rand(p_ideal.shape, generator=rng_gen, device=device) - 1)

    return p_ideal + eps_p

def generateUniform(min_val, max_val, out_shape, rng_gen=None, device='cuda'):
    '''Tensor of shape out_shape with entries ~ U[min_val, max_val).'''
    return torch.rand(out_shape, generator=rng_gen, device=device) * (max_val - min_val) + min_val

def generateGPImpairments(min_gain, max_gain, min_phase, max_phase, K, rng_gen=None, device='cuda'):
    '''GPIs gamma_x with |[gamma_x]_k| ~ U[min_gain, max_gain] and phase ~ U[min_phase, max_phase].
    Shape: (K,1)'''
    gain = generateUniform(min_gain, max_gain, (K,1), rng_gen, device)
    phase = generateUniform(min_phase, max_phase, (K,1), rng_gen, device)
    gamma = gain*torch.exp(1j*phase)

    return gamma

def generateInterval(mean_min, mean_max, span_min, span_max, batch_size=1, rng_gen=None, device='cuda'):
    '''Random intervals [mean - span/2, mean + span/2] with mean ~ U[mean_min, mean_max] and
    span ~ U[span_min, span_max]. Returns (minimum, maximum), each of shape (batch_size,1).'''
    mean = generateUniform(mean_min, mean_max, (batch_size, 1), rng_gen, device=device)
    span = generateUniform(span_min, span_max, (batch_size, 1), rng_gen, device=device)
    minimum = mean - span / 2.0
    maximum = mean + span / 2.0

    return minimum, maximum

def noise(var, dims, rng_gen=None, device='cuda'):
    '''Circularly-symmetric complex Gaussian samples CN(var) of shape dims.'''
    return torch.sqrt(var/2.0) * (torch.randn(dims, generator=rng_gen, device=device) + 1j * torch.randn(dims, generator=rng_gen, device=device))

def steeringMatrix(theta_min, theta_max, p, lamb, gamma=None, rng_gen=None):
    '''Impaired steering vectors a(theta; gamma, p) of eqs. (11)/(14) as columns, with
    theta ~ U[theta_min, theta_max) element-wise (pass theta_min = theta_max for fixed angles).
    Returns the (K, N) matrix and the drawn angles of shape (N,1).'''
    if gamma is None:
        gamma = torch.ones((len(p),1), dtype=torch.cfloat, device = p.device)

    theta = generateUniform(theta_min, theta_max, theta_min.shape, rng_gen=rng_gen, device=theta_min.device)
    theta = theta.view(1, -1)

    matrix = gamma * torch.exp(-1j * 2 * np.pi / lamb * p.view(-1,1).type(torch.cfloat) @ torch.sin(theta.type(torch.cfloat)))
    return matrix, theta.view(-1,1)

def delayMatrix(R_min, R_max, delta_f, S, rng_gen=None):
    '''Frequency-domain steering vectors rho(tau) of eq. (4) as columns, with round-trip delay
    tau = 2R/c and R ~ U[R_min, R_max) element-wise.
    Returns the (S, N) matrix and the drawn ranges of shape (N,1).'''
    device = R_min.device
    R = generateUniform(R_min, R_max, R_min.shape, rng_gen, device=device)
    R = R.view(1, -1)

    matrix = torch.exp(-1j*2*np.pi*delta_f * 2/3e8 * torch.arange(0,S, device=device).view(-1,1).type(torch.cfloat) @ R.type(torch.cfloat))
    return matrix, R.view(-1,1)

def createPrecoder(theta_min, theta_max, A_tx, N_theta, P, device='cuda'):
    '''Precoder f_x of eq. (13) (scaled to power P) illuminating [theta_min, theta_max]. A_tx holds
    the steering vectors of the N_theta-point grid over [-pi/2, pi/2]. If the sector contains no
    grid point, the closest one is used. Shape: (batch_size, K, 1)'''
    batch_size = theta_min.shape[0] if theta_min.dim()>0 else 1
    angle_vector = torch.linspace(-np.pi/2, np.pi/2, N_theta, device = device).reshape(1,-1)
    b_vec = ((angle_vector >= theta_min) & (angle_vector <= theta_max))*1.0            #Shape: (batch_size, N_theta)

    zero_indices = b_vec.sum(dim=1) == 0
    diff = torch.abs(theta_min - angle_vector)
    min_indices = torch.argmin(diff, dim=1)
    b_vec[zero_indices, min_indices[zero_indices]] = 1.0
    b_vec = b_vec.type(torch.cfloat).reshape(batch_size, N_theta, 1)
    f_x = A_tx.conj() @ b_vec
    return f_x * torch.sqrt(P) / torch.norm(f_x, dim=-2, keepdim=True)

def isacPrecoder(f_s, f_c, omega_r, phi, P):
    '''ISAC precoder f_tx of eq. (10), with an additional phase phi on f_c.'''
    f_tx = torch.sqrt(omega_r) * f_s + torch.sqrt(1-omega_r) * torch.exp(1j * phi) * f_c     #Shape: (batch_size, K, 1)
    return f_tx * torch.sqrt(P) / torch.norm(f_tx, dim=-2, keepdim=True)

def computeCommMSE(y_c, kappa, x_ref):
    '''|[y_c]_s - [kappa]_s x|^2 for every subcarrier and candidate symbol x in x_ref.
    Shape: (batch_size*S, |X|)'''
    y_c_flatten = y_c.reshape(-1,1)
    kappa_flatten = kappa.reshape(-1,1)
    return torch.abs(y_c_flatten - kappa_flatten*x_ref.reshape(1,-1))**2


#######################################################################
# Sensing functions                                                   #
#######################################################################
def _reflections(system, theta_min, theta_max, R_min, R_max, f_tx, x, num, drawGain, rng_gen):
    '''Noiseless sum over num[b] scatterers of alpha a_rx(theta) a_tx(theta)^T f_tx [x * rho(tau)]^T,
    with angles and ranges drawn in each sample's sector. drawGain(R, n) returns the n complex
    gains given the scatterer ranges R. Returns the sum (batch_size, K, S), theta, R and alpha.'''
    K, S = system.K, system.S
    device = f_tx.device
    batch_size = len(num)
    #One row per scatterer, summed back per sample at the end
    n_total = num.sum()
    f_tx = torch.repeat_interleave(f_tx, num, dim=0)
    x = torch.repeat_interleave(x, num, dim=0)
    #Limits that differ across samples are interpreted as one sector per sample
    if not torch.all(theta_min == theta_max):
        theta_min = torch.repeat_interleave(theta_min, num, dim=0)
        theta_max = torch.repeat_interleave(theta_max, num, dim=0)
    if not torch.all(R_min == R_max):
        R_min = torch.repeat_interleave(R_min, num, dim=0)
        R_max = torch.repeat_interleave(R_max, num, dim=0)

    rho, R = delayMatrix(R_min, R_max, system.delta_f, S, rng_gen)
    rho = rho.transpose(-1,-2).view(n_total, S, 1)
    R = R.view(n_total, 1, 1)
    alpha = drawGain(R, n_total)

    a_tx, theta = steeringMatrix(theta_min, theta_max, system.p_tx, system.lamb, system.gamma_tx, rng_gen)
    a_rx, _ = steeringMatrix(theta, theta, system.p_rx, system.lamb, system.gamma_rx, rng_gen)
    a_tx = a_tx.transpose(-1,-2).reshape(n_total,K,1)
    a_rx = a_rx.transpose(-1,-2).reshape(n_total,K,1)

    reflection = alpha * a_rx @ a_tx.permute(0,2,1) @ f_tx @ (x * rho).permute(0,2,1)     #Shape: (n_total, K, S)
    ind = torch.arange(batch_size, device=device).repeat_interleave(num)
    reflection_sum = torch.index_add(torch.zeros((batch_size,K,S), dtype=torch.cfloat, device=device),0,ind, reflection)
    return reflection_sum, theta, R, alpha

def sensingChannel(system, theta_min, theta_max, R_min, R_max, f_tx, x, T, rng_gen=None, C=None, sigma_c2=None):
    '''Sensing observations Y_s of eq. (1) for T[b] targets with gains from eq. (2) and, if C is
    given, C[b] clutter scatterers with gains ~ CN(sigma_c2) in the same sector.
    Returns Y_s (batch_size, K, S) and the true target angles, ranges and gains alpha.'''
    device = f_tx.device

    def targetGain(R, n):
        rcs = torch.empty((n,1,1), device=device).exponential_(lambd=1/system.sigma_mean, generator=rng_gen)
        mag = torch.sqrt(rcs*system.lamb**2/((4*np.pi)**3*R**4))
        phase = generateUniform(0, 2*np.pi, (n,1,1), rng_gen, device=device)
        return mag*torch.exp(1j*phase)

    Y_s, theta, R, alpha = _reflections(system, theta_min, theta_max, R_min, R_max, f_tx, x, T, targetGain, rng_gen)
    if C is not None:
        clutter, _, _, _ = _reflections(system, theta_min, theta_max, R_min, R_max, f_tx, x, C,
                                        lambda R, n: noise(sigma_c2, (n,1,1), rng_gen, device=device), rng_gen)
        Y_s = Y_s + clutter

    W = noise(system.noise_var, Y_s.shape, rng_gen, device=device)
    Y_s = Y_s + W

    return Y_s, theta.view(-1,1), R.view(-1,1), alpha


def omp(Y_s, theta_min, theta_max, R_min, R_max, delta_f, I, Phi_a, N_theta, N_tau):
    '''I iterations of the OMP of Algorithm 1 (without the termination threshold) over an
    N_theta x N_tau angle-delay grid of [theta_min, theta_max] x [R_min, R_max].
    Phi_a is the angular dictionary of eq. (15), shape (K, N_theta) or (batch_size, K, N_theta).
    Y_s is assumed to have the symbols x removed, so that Phi_d reduces to delay steering vectors.
    Returns, for each iteration, the maximum of the ADM |Phi_a^H Y_s^(i) Phi_d^*| and the
    estimated angle and range, each of shape (batch_size, I), and the residual Y_s^(I) of eq. (22).'''
    device = Y_s.device
    batch_size, K, S = Y_s.shape

    #Number of angular and range sectors (either 1 or batch_size)
    num_angle_sectors = 1 if ((not torch.is_tensor(theta_min)) or (theta_min.dim()==0)) else len(theta_min)
    num_angle_sectors_arange = torch.arange(num_angle_sectors, device=device)
    num_range_sectors = 1 if ((not torch.is_tensor(R_min)) or (R_min.dim()==0)) else len(R_min)
    num_range_sectors_arange = torch.arange(num_range_sectors, device=device)
    if Phi_a.dim() == 2:
        Phi_a = Phi_a.unsqueeze(0).expand(num_angle_sectors, -1, -1)

    #Angle and range grids, and delay dictionary
    delta_angle = (theta_max-theta_min)/(N_theta-1)
    theta_grid_matrix = theta_min + delta_angle*torch.arange(N_theta, device=device).reshape(1,-1)  #Shape: (num_angle_sectors, N_theta)
    delta_range = (R_max-R_min)/(N_tau-1)
    range_grid_matrix = R_min + delta_range*torch.arange(N_tau, device=device).reshape(1,-1)        #Shape: (num_range_sectors, N_tau)
    Phi_d, _ = delayMatrix(range_grid_matrix, range_grid_matrix, delta_f, S)
    Phi_d = Phi_d.transpose(-1,-2).reshape(num_range_sectors, N_tau, S).permute(0,2,1)

    theta_hat, R_hat = torch.empty(batch_size, I, device=device), torch.empty(batch_size, I, device=device)
    adm_max = torch.empty(batch_size, I, device=device)
    #Atoms selected so far, vectorized (empty 3-D tensor so that the first cat is well defined)
    M_vec = torch.empty((batch_size, K*S, 0), device=device, dtype=torch.cfloat)
    residual = torch.clone(Y_s)
    Y_s_vec = Y_s.view(batch_size, K*S, 1)
    for i in range(I):
        adm = torch.abs( torch.conj(Phi_a.transpose(-1,-2)) @ residual @ torch.conj(Phi_d) )     #Shape: (batch_size, N_theta, N_tau)

        #Angle-delay detection, eq. (18)
        maximum = adm.view(batch_size, -1).argmax(dim=-1)
        max_theta = torch.div(maximum, N_tau, rounding_mode='floor')
        max_range = maximum % N_tau
        est_theta = theta_grid_matrix[num_angle_sectors_arange, max_theta].view(batch_size, 1)
        est_range = range_grid_matrix[num_range_sectors_arange, max_range].view(batch_size, 1)
        adm_max_i, _ = torch.max(adm.view(batch_size, -1), dim=1)

        #Atom update, eqs. (19)-(20)
        est_a = Phi_a[num_angle_sectors_arange.unsqueeze(1), :, max_theta.unsqueeze(1)].reshape((batch_size, K, 1))
        est_rho, _ = delayMatrix(est_range, est_range, delta_f, S)
        est_rho = est_rho.transpose(0,1).view((batch_size, S, 1))
        M = est_a @ est_rho.permute(0,2,1)   #Shape: (batch_size, K, S)
        M_vec = torch.cat((M_vec, M.view(batch_size,K*S,1)), dim=-1)

        #Gain estimates, eq. (21), and residual, eq. (22)
        with torch.no_grad():
            alpha_hat = torch.linalg.lstsq(M_vec, Y_s_vec).solution
        residual = (Y_s_vec - M_vec@alpha_hat).view(batch_size, K, S)

        theta_hat[:,i] = est_theta.flatten()
        R_hat[:,i] = est_range.flatten()
        adm_max[:,i] = adm_max_i.flatten()

    return adm_max, theta_hat, R_hat, residual


#######################################################################
# Set metric                                                          #
#######################################################################
def gospa(estimation, target, gamma, mu=2, p=2, device='cuda'):
    '''GOSPA of eq. (38) between two sets given as tensors of shape (n, dims), or
    (batch_size, n, dims) to evaluate several pairs in parallel.
    Returns the GOSPA and the mean localization error of the assigned pairs (without
    cut-off or cardinality penalty).'''
    #The estimation is always taken as the set with lower cardinality
    if estimation.numel() > target.numel():
        estimation, target = target, estimation

    if estimation.dim() == 1:
        estimation = estimation.unsqueeze(0).unsqueeze(-1)
        target = target.unsqueeze(0).unsqueeze(-1)
    elif estimation.dim() == 2:
        estimation = estimation.unsqueeze(0)
        target = target.unsqueeze(0)

    b_size, card_est, dimensions = estimation.shape
    card_tgt = target.shape[-2]

    if card_est > 0:
        #All assignments: permutations of the estimates against combinations of the targets
        num_permutations_est = math.factorial(card_est)
        permutation_list = list(itertools.permutations(range(card_est)))
        permutation_index = torch.tensor(permutation_list, dtype=torch.long, device=device)
        estimation_comb = estimation[:,permutation_index,:].transpose(-2,-1).reshape(b_size,num_permutations_est,dimensions,1,card_est)

        combinations_tgt = torch.combinations(torch.arange(card_tgt, device=device), card_est)
        num_combinations_tgt = combinations_tgt.shape[0]
        tgt_comb = target[:,combinations_tgt].permute(0,3,1,2).reshape(b_size,dimensions,num_combinations_tgt,card_est)
        tgt_comb = tgt_comb[:,None,:,:]

        dist = torch.norm(estimation_comb-tgt_comb,dim=-3)     #Shape: (batch_size, num_permutations_est, num_combinations_tgt, card_est)
        cut_off = torch.minimum(dist,gamma)
        dist_sum = torch.sum(cut_off**p,dim=-1)
        min_dist, _ = torch.min(dist_sum.reshape(b_size,-1), dim=-1)
        avg_dist, _ = torch.min(torch.sum(dist**p, dim=-1).reshape(b_size,-1), dim=-1)
    else:
        min_dist = torch.zeros((b_size),device=device)
        avg_dist = float('nan')*torch.ones((b_size), device=device)

    gospa_out = (min_dist + gamma**p/mu*(card_tgt-card_est))**(1/p)
    return gospa_out, avg_dist**(1/p)


#######################################################################
# Communication functions                                             #
#######################################################################
def MPSK(M, rotation=0, device='cuda'):
    '''Symbols of an M-PSK constellation rotated by `rotation` [rad].'''
    return torch.exp(1j * (2*np.pi/M * torch.arange(0,M, device=device) + rotation))

def commChannel(system, theta_min_c, theta_max_c, R_min_c, R_max_c, f_tx, x, T_c, rng_gen=None, device='cuda'):
    '''Received communication signal y_c of eq. (6) and CSI kappa of eq. (8), each of shape
    (batch_size, S, 1), for T_c[b] paths. Path 1 is the LoS (gain from eq. (7)); the scatterers
    lie in the UE sector, at least 1 deg away from the LoS and with a delay spread within the
    cyclic prefix.'''
    K, S, lamb, delta_f, R_cp = system.K, system.S, system.lamb, system.delta_f, system.R_cp
    batch_size = len(T_c)

    #One row per path, summed back per sample at the end. All paths are first drawn as
    #NLoS and the LoS gains, angles and ranges are then written at the first path of each sample.
    n_total = T_c.sum()
    f_rep = torch.repeat_interleave(f_tx, T_c, dim=0)
    x_rep = torch.repeat_interleave(x, T_c, dim=0)
    if theta_min_c.numel() > 1:
        theta_min_c = theta_min_c.reshape(batch_size,1,1)
    if theta_max_c.numel() > 1:
        theta_max_c = theta_max_c.reshape(batch_size,1,1)
    if R_min_c.numel() > 1:
        R_min_c = R_min_c.reshape(batch_size,1,1)
    if R_max_c.numel() > 1:
        R_max_c = R_max_c.reshape(batch_size,1,1)

    #LoS range R_1 and angle
    R_los = torch.rand((batch_size,1,1), generator=rng_gen, device=device) * (R_max_c - R_min_c) + R_min_c
    R_los_rep = R_los.repeat_interleave(T_c,dim=0)
    theta_los = torch.rand((batch_size,1,1), generator=rng_gen, device=device) * (theta_max_c - theta_min_c) + theta_min_c
    theta_los_rep = theta_los.repeat_interleave(T_c,dim=0)
    #NLoS angles, at least 1 deg away from the LoS
    theta_min_scat = torch.repeat_interleave(theta_min_c.reshape(batch_size,1,1), T_c, dim=0)
    theta_max_scat = torch.repeat_interleave(theta_max_c.reshape(batch_size,1,1), T_c, dim=0)
    theta_nlos = torch.rand((n_total,1,1), generator=rng_gen, device=device) * (theta_max_scat - theta_min_scat) + theta_min_scat
    angle_diff = theta_los_rep-theta_nlos
    min_sep = torch.tensor(1*np.pi/180.0, device=device)
    angle_mask = torch.abs(angle_diff) < min_sep
    theta_nlos[angle_mask] = theta_los_rep[angle_mask] + (torch.randint(0,2, theta_los_rep[angle_mask].shape, generator=rng_gen,device=device)*2-1)*min_sep
    angle_diff = theta_los_rep-theta_nlos
    #Maximum BS-scatterer range such that the delay spread stays within the cyclic prefix
    R_nlos_max = R_cp * (R_cp/2.0 + R_los_rep)/(R_los_rep*(1-torch.cos(angle_diff))+R_cp)
    if torch.isnan(R_nlos_max).sum() > 0:
        print(f'R_nlos_max: {R_nlos_max}')
        raise ValueError('R_nlos_max has a nan')
    #BS-scatterer range R_t1 and scatterer-UE range R_t2 (law of cosines)
    R_min_scat = torch.repeat_interleave(R_min_c.reshape(batch_size,1,1), T_c, dim=0)
    R_t1 = torch.rand((n_total,1,1), generator=rng_gen, device=device) * (R_nlos_max - R_min_scat) + R_min_scat
    R_t2 = torch.sqrt(R_los_rep**2 + R_t1**2 - 2*R_los_rep*R_t1*torch.cos(angle_diff))
    R_nlos = R_t1 + R_t2
    #NLoS gains, eq. (7)
    rcs_nlos = torch.empty((n_total,1,1), device=device).exponential_(lambd=1/system.sigma_mean, generator=rng_gen)
    mag_nlos = torch.sqrt(rcs_nlos*lamb**2/((4*np.pi)**3*R_t1**2*R_t2**2))
    if torch.isnan(mag_nlos).sum() > 0:
        print(f'mag_nlos: {mag_nlos}')
        raise ValueError('mag_nlos has a nan')
    phase_nlos = torch.rand((n_total,1,1), generator=rng_gen, device=device)*2*np.pi
    alpha = mag_nlos*torch.exp(1j*phase_nlos)

    #LoS gains, eq. (7)
    R_los = R_los[T_c > 0]
    theta_los = theta_los[T_c > 0]
    mag_los = torch.sqrt(lamb**2/(4*np.pi*R_los)**2)
    phase_los = torch.rand((len(R_los),1,1), generator=rng_gen, device=device)*2*np.pi
    alpha_los = mag_los*torch.exp(1j*phase_los)

    #Write the LoS path at the first path of each sample
    insert_idx = torch.unique(torch.cat((torch.tensor([0], device=device), torch.cumsum(T_c,dim=0))))[:-1]
    alpha[insert_idx] = alpha_los
    theta_nlos[insert_idx] = theta_los; theta = theta_nlos
    R_nlos[insert_idx] = R_los; R = R_nlos

    a_tx, _ = steeringMatrix(theta, theta, system.p_tx, lamb, system.gamma_tx)
    a_tx = a_tx.transpose(1,0).view(n_total,K,1)
    rho = torch.exp(-1j * 4 * np.pi / 3e8 * delta_f * R.type(torch.cfloat) * torch.arange(S, device=device).reshape(1,S,1).type(torch.cfloat))

    #Received signal, eq. (6)
    y_noiseless = alpha * torch.bmm(a_tx.permute(0,2,1), f_rep) * (x_rep * rho)  #Shape: (n_total, S, 1)
    ind = torch.arange(batch_size, device=device).repeat_interleave(T_c)
    y_noiseless_sum = torch.index_add(torch.zeros((batch_size,S,1), dtype=torch.cfloat, device=device),0,ind, y_noiseless)
    w = noise(system.noise_var, (batch_size,S,1), rng_gen, device=device)
    y_c = y_noiseless_sum + w

    #CSI, eq. (8)
    kappa = alpha * torch.bmm(a_tx.permute(0,2,1), f_rep) * rho
    kappa = torch.index_add(torch.zeros((batch_size,S,1), dtype=torch.cfloat, device=device),0,ind, kappa)

    return y_c, kappa

def MLdecoder(y_c, kappa, x_ref):
    '''Subcarrier-wise ML decoder of eq. (25). Returns the indices of the estimated symbols
    in x_ref. Shape: (batch_size, S)'''
    metric = torch.abs(y_c - kappa * x_ref.reshape(1,len(x_ref)))**2
    return torch.argmin(metric, dim=-1)


#######################################################################
# Training functions                                                  #
#######################################################################
def setReferenceSensor(network, ref_idx, p_tx, p_rx, gamma_tx, gamma_rx):
    '''Fix the element ref_idx of both ULAs to its true value and never learn it (eq. (35)):
    its parameters are set to Xi, its gradient is masked and projectImpairments leaves it
    untouched.'''
    K = network.omega_tx.shape[0]
    if not 0 <= ref_idx < K:
        raise ValueError(f'The reference sensor index must be in [0, {K-1}], got {ref_idx}')

    network.ref_idx = ref_idx

    network.omega_tx.data[ref_idx] = p_tx[ref_idx].detach()
    network.omega_rx.data[ref_idx] = p_rx[ref_idx].detach()
    network.xi_tx.data[ref_idx] = gamma_tx[ref_idx].detach()
    network.xi_rx.data[ref_idx] = gamma_rx[ref_idx].detach()

    #The hook lives on the Parameter, so it survives the .data rebinding of projectImpairments
    def _maskRefGrad(grad):
        grad_masked = grad.clone()
        grad_masked[ref_idx] = 0
        return grad_masked
    for param in (network.omega_tx, network.omega_rx, network.xi_tx, network.xi_rx):
        param.register_hook(_maskRefGrad)

def _sortPositions(omega, ref_idx=None):
    '''Sort the antenna positions. The reference sensor (if any) keeps its slot and only the
    learned positions are sorted among themselves.'''
    if ref_idx is None:
        omega_sorted, _ = torch.sort(omega, dim=0)
        return omega_sorted
    learned = torch.ones(omega.shape[0], dtype=torch.bool, device=omega.device)
    learned[ref_idx] = False
    omega_sorted = omega.clone()
    omega_sorted[learned], _ = torch.sort(omega[learned], dim=0)
    return omega_sorted

def _capGainPhase(xi, one, ref_idx=None):
    '''Cap |[xi]_k| at 1 keeping the phase. The reference sensor (if any) is left untouched.'''
    xi_capped = torch.minimum(torch.abs(xi), one)*torch.exp(1j*torch.angle(xi))
    if ref_idx is not None:
        xi_capped[ref_idx] = xi[ref_idx]
    return xi_capped

def projectImpairments(network, device='cuda'):
    '''Projection of Psi onto the feasible set of eq. (27) (eq. (35) with a reference sensor),
    applied in place after every optimizer step.'''
    ref_idx = getattr(network, 'ref_idx', None)
    network.omega_tx.data = _sortPositions(network.omega_tx.data, ref_idx)
    network.omega_rx.data = _sortPositions(network.omega_rx.data, ref_idx)
    one = torch.tensor(1, device=device)
    network.xi_tx.data = _capGainPhase(network.xi_tx.data, one, ref_idx)
    network.xi_rx.data = _capGainPhase(network.xi_rx.data, one, ref_idx)

def perturbPrecoder(f_tx, sigma, device='cuda'):
    '''Perturbed precoder f_tilde = sqrt(1-sigma^2) f_tx + eps_f, eps_f ~ CN(sigma^2 I),
    normalized to unit norm (Sec. IV-A).'''
    f_tilde = f_tx*np.sqrt(1-sigma**2) + sigma/np.sqrt(2)*(torch.randn(f_tx.shape, device=device) + 1j*torch.randn(f_tx.shape, device=device))
    return f_tilde/torch.norm(f_tilde, dim=-2, keepdim=True)

def logPdfPerturbation(f_tilde, f_tx, sigma):
    '''Gaussian log-density log p_{f_tx,sigma}(f_tilde) used in the gradient of eq. (34).
    Shape: (batch_size, 1)'''
    mean = f_tilde - np.sqrt(1-sigma**2)*f_tx
    K = f_tilde.shape[1]
    return (-torch.norm(mean, dim=1)**2 / sigma**2 - K*np.log(np.pi * sigma**2)).view(-1,1)

def _perturbSector(theta_min, theta_max, sigma_theta, rng_gen, device):
    '''Assumed sector of Sec. IV-H: each endpoint plus N(0, sigma_theta^2), clamped to
    [-pi/2, pi/2] and re-sorted so that min <= max.'''
    theta_min_p = theta_min + sigma_theta*torch.randn(theta_min.shape, generator=rng_gen, device=device)
    theta_max_p = theta_max + sigma_theta*torch.randn(theta_max.shape, generator=rng_gen, device=device)
    theta_min_p = torch.clamp(theta_min_p, -np.pi/2, np.pi/2)
    theta_max_p = torch.clamp(theta_max_p, -np.pi/2, np.pi/2)
    return torch.minimum(theta_min_p, theta_max_p), torch.maximum(theta_min_p, theta_max_p)

def _drawTrainingScenario(supervised, t, system, B, A_tx, omega_r, phi, sens, comm, rng_gen, device,
                          sigma_theta=None):
    '''Draw one training minibatch: number of targets and paths, symbols, sensing and
    communication sectors and the ISAC precoder f_tx. With supervised=True the number of targets
    is t+1 for the whole batch. If sigma_theta is given, f_tx and the returned OMP sector are
    built from the perturbed (assumed) sectors, while the channel uses the true ones.'''
    if supervised:
        T = (t+1) * torch.ones((B, ), dtype=torch.int, device=device)
    else:
        #Drawn from the global RNG (not rng_gen), as in the original experiments
        T = torch.randint(0,system.T_max+1,(B,), device=device)
    T_c = torch.randint(1,system.T_max_c+1,(B,), generator=rng_gen, device=device)

    msg = torch.randint(0,len(system.x_ref),size=(B*system.S,), dtype=torch.int64, generator=rng_gen, device=device)
    x = system.x_ref[msg].reshape(B, system.S, 1)

    theta_min_s, theta_max_s = generateInterval(sens.theta_mean_min, sens.theta_mean_max, sens.delta_theta_min, sens.delta_theta_max, B, rng_gen=rng_gen, device=device)
    theta_min_c, theta_max_c = generateInterval(comm.theta_mean_min, comm.theta_mean_max, comm.delta_theta_min, comm.delta_theta_max, B, rng_gen=rng_gen, device=device)
    R_min_s, R_max_s = generateInterval(sens.R_mean_min, sens.R_mean_max, sens.delta_R_min, sens.delta_R_max, B, rng_gen=rng_gen, device=device)
    R_min_c, R_max_c = generateInterval(comm.R_mean_min, comm.R_mean_max, comm.delta_R_min, comm.delta_R_max, B, rng_gen=rng_gen, device=device)

    if sigma_theta is not None:
        theta_min_s_a, theta_max_s_a = _perturbSector(theta_min_s, theta_max_s, sigma_theta, rng_gen, device)
        theta_min_c_a, theta_max_c_a = _perturbSector(theta_min_c, theta_max_c, sigma_theta, rng_gen, device)
    else:
        theta_min_s_a, theta_max_s_a = theta_min_s, theta_max_s
        theta_min_c_a, theta_max_c_a = theta_min_c, theta_max_c

    f_s = createPrecoder(theta_min_s_a, theta_max_s_a, A_tx, system.N_theta, system.P, device=device)
    f_c = createPrecoder(theta_min_c_a, theta_max_c_a, A_tx, system.N_theta, system.P, device=device)
    f_tx = isacPrecoder(f_s, f_c, omega_r, phi, system.P)

    return (T, T_c, msg, x,
            theta_min_s, theta_max_s, theta_min_c, theta_max_c,
            R_min_s, R_max_s, R_min_c, R_max_c, f_tx,
            theta_min_s_a, theta_max_s_a)

def _slSensingLoss(network, theta, R, Y_s, T, system):
    '''Supervised sensing loss of eq. (40) without the square, i.e. the negative magnitude of the
    ADM at the true target positions, averaged over all targets of the batch.'''
    a_rx, _ = steeringMatrix(theta, theta, network.omega_rx, system.lamb, network.xi_rx)
    rho, _ = delayMatrix(R, R, system.delta_f, system.S)
    rho = rho.transpose(1,0).reshape(-1, system.S,1)
    Y_s_rep = torch.repeat_interleave(Y_s, T, dim=0)
    adm_true = torch.abs(a_rx.conj().transpose(1,0).reshape(-1,1,system.K) @ Y_s_rep @ rho.conj())
    return -torch.mean(adm_true.reshape(T.sum()))

def _cceLoss(y_c, kappa, x_ref, msg):
    '''Supervised communication loss of eqs. (41)-(42), with log10 in place of log.'''
    comm_mse = computeCommMSE(y_c, kappa, x_ref)
    #Avoid log(0) for exact matches
    mask = (comm_mse > 0).all(dim=-1)
    neg_mse = -torch.log10(comm_mse[mask])
    return torch.mean(F.cross_entropy(neg_mse, msg[mask], reduction='none'))


def train(network, method, system, sens, comm, B, n_it, omega_r, eta_r, optimizer, scheduler,
          optimizer_rx=None, scheduler_rx=None, sigma=0.0, sens_loss='adm', I_omp=1,
          tx_objective='isac', train_tx_mask=None, train_rx_mask=None,
          C_max=None, sigma_c2=0.0, sigma_theta=None, rng_gen=None, device='cuda'):
    '''
    Calibration of Psi = network by POGD (Sec. III-D), for n_it iterations of T_max minibatches
    of B samples each. omega_r and eta_r hold one value per iteration. Methods:
        - 'gf_ul'    : proposed GF UL. Unsupervised losses (23)/(24) and (26) with the
                       gradient-free TX update of eq. (34) (precoder perturbed with std sigma).
                       Separate TX (optimizer) and RX (optimizer_rx) optimizers; the RX is not
                       stepped when eta_r = 0.
        - 'slcb'     : SLCB. Supervised losses (40)-(41) with the true channel gradient.
        - 'slcb_pert': SLCB with the perturbed precoder f_tilde of std sigma.
    GF UL options:
        - sens_loss: 'adm' (eq. (23)) or 'residual' (eq. (24) with I_omp OMP iterations).
        - tx_objective: 'isac' drives the TX with the eta_r-weighted ISAC loss and 'comm' with
          the communication loss only (alternate optimization of Sec. IV-E2).
        - train_tx_mask, train_rx_mask: boolean masks over the iterations where each side is
          trained (None: all).
        - C_max, sigma_c2: if C_max is given, C ~ U{0, ..., C_max} clutter scatterers of
          power sigma_c2 are added to the sensing channel (Sec. IV-D2).
        - sigma_theta: std [rad] of the angular prior perturbation (Sec. IV-H), or None.
    Returns the per-iteration sensing and communication losses and the training SCNR [dB].
    '''
    if method not in ('gf_ul', 'slcb', 'slcb_pert'):
        raise ValueError(f"Unknown method '{method}'")
    if tx_objective not in ('isac', 'comm'):
        raise ValueError(f"Unknown TX objective '{tx_objective}'")

    print('*** Training started ***')
    network.train()

    T_max, K, S = system.T_max, system.K, system.S
    lamb, N_theta, N_tau = system.lamb, system.N_theta, system.N_tau
    supervised = method != 'gf_ul'

    loss_r_list, loss_c_list, scnr = [], [], []

    if train_tx_mask is None:
        train_tx_mask = torch.ones(n_it, dtype=torch.bool, device=device)
    if train_rx_mask is None:
        train_rx_mask = torch.ones(n_it, dtype=torch.bool, device=device)

    phi_zero = torch.tensor(0, device=device)
    ones_K = torch.ones((K,1), device=device).type(torch.cfloat)

    for it in range(n_it):
        loss_r_t = torch.zeros(T_max, device=device)
        loss_c_t = torch.zeros(T_max,device=device)
        if method == 'gf_ul':   #Policy-weighted sensing loss that updates the TX
            loss_r_pol_t = torch.zeros(T_max, device=device)

        A_tx, _ = steeringMatrix(system.theta_grid, system.theta_grid, network.omega_tx, lamb, network.xi_tx)

        scnr_it = []
        for t in range(T_max):
            (T, T_c, msg, x,
             theta_min_s, theta_max_s, theta_min_c, theta_max_c,
             R_min_s, R_max_s, R_min_c, R_max_c, f_tx,
             theta_min_s_a, theta_max_s_a) = _drawTrainingScenario(
                supervised, t, system, B, A_tx, omega_r[it], phi_zero,
                sens, comm, rng_gen, device, sigma_theta=sigma_theta)
            C = torch.randint(0, C_max+1, (B,), generator=rng_gen, device=device) if C_max is not None else None

            # ================= CHANNEL ================= #
            if method == 'gf_ul':
                with torch.no_grad():
                    f_tilde = perturbPrecoder(f_tx, sigma, device)
                    Y_s, theta, R, alpha = sensingChannel(system, theta_min_s, theta_max_s, R_min_s, R_max_s,
                                                          f_tilde, x, T, rng_gen, C, sigma_c2)
                    y_c, kappa = commChannel(system, theta_min_c, theta_max_c, R_min_c, R_max_c,
                                             f_tilde, x, T_c, rng_gen, device)
            elif method == 'slcb_pert':
                f_tilde = perturbPrecoder(f_tx, sigma, device)
                #No gradient through the sensing channel, so that sensing does not update the TX
                with torch.no_grad():
                    Y_s, theta, R, alpha = sensingChannel(system, theta_min_s, theta_max_s, R_min_s, R_max_s,
                                                          f_tilde, x, T, rng_gen)
                y_c, kappa = commChannel(system, theta_min_c, theta_max_c, R_min_c, R_max_c,
                                         f_tilde, x, T_c, rng_gen, device)
            else:
                Y_s, theta, R, alpha = sensingChannel(system, theta_min_s, theta_max_s, R_min_s, R_max_s,
                                                      f_tx, x, T, rng_gen)
                y_c, kappa = commChannel(system, theta_min_c, theta_max_c, R_min_c, R_max_c,
                                         f_tx, x, T_c, rng_gen, device)

            #Remove the known symbols x from the sensing observations
            Y_s /= ones_K @ x.permute(0,2,1)

            # ================= LOSSES ================= #
            if supervised:
                loss_c_t[t] = _cceLoss(y_c, kappa, system.x_ref, msg)
                loss_r_t[t] = _slSensingLoss(network, theta, R, Y_s, T, system)
            else:
                log_pdf = logPdfPerturbation(f_tilde, f_tx, sigma)
                #Eq. (26), weighted by the log-density for eq. (34)
                loss_c_t[t] = -torch.mean(torch.norm(y_c, dim=-2)*log_pdf)

                #Angular dictionary of eq. (15) over the (assumed) sensing sector
                delta_angle = (theta_max_s_a-theta_min_s_a)/(N_theta-1)
                theta_grid_matrix = theta_min_s_a + delta_angle*torch.arange(N_theta, device=device).reshape(1,-1)  #Shape: (B, N_theta)
                Phi_a, _ = steeringMatrix(theta_grid_matrix, theta_grid_matrix, network.omega_rx, lamb, network.xi_rx)
                Phi_a = Phi_a.transpose(-1,-2).reshape(B, N_theta, K).permute(0,2,1)

                #The detached losses weighted by the log-density update the TX (eq. (34)) and
                #the plain losses update the RX
                if sens_loss == 'adm':
                    adm_max, _, _, _ = omp(Y_s, theta_min_s_a, theta_max_s_a, R_min_s, R_max_s, system.delta_f, 1, Phi_a, N_theta, N_tau)
                    adm_max = adm_max[:, 0]
                    loss_r_t[t] = -torch.mean(adm_max)
                    loss_r_pol_t[t] = -torch.mean(adm_max.detach()*log_pdf)
                elif sens_loss == 'residual':
                    _, _, _, residual = omp(Y_s, theta_min_s_a, theta_max_s_a, R_min_s, R_max_s, system.delta_f, I_omp, Phi_a, N_theta, N_tau)
                    loss_r_t[t] = torch.mean(torch.linalg.norm(residual, dim=(-2,-1)))
                    loss_r_pol_t[t] = torch.mean(torch.linalg.norm(residual.detach(), dim=(-2,-1))*log_pdf.flatten())

            #Empirical SCNR of eq. (5) (skipped if the minibatch has no targets)
            if alpha.numel() > 0:
                scnr_it.append(torch.mean(torch.abs(alpha)**2)/(system.noise_var/(system.P*K) + sigma_c2))

        loss_r = torch.mean(loss_r_t)
        loss_c = torch.mean(loss_c_t)
        if method == 'gf_ul':
            loss_r_pol = torch.mean(loss_r_pol_t)
        if it % 2000 == 0:
            print(f'Sensing loss on iteration {it}: {loss_r}', flush=True)
            print(f'Communication loss on iteration {it}: {loss_c}', flush=True)

        scnr_it = [s.item() for s in scnr_it]
        if any(scnr_it):
            scnr.append(np.mean(scnr_it))

        # ================= BACKWARD ================= #
        if method == 'gf_ul':
            #A side whose optimizer is not stepped keeps its parameters exactly frozen (Adam
            #would otherwise keep moving them with the accumulated momentum)
            train_rx = bool(train_rx_mask[it] and eta_r[it] > 0)
            if train_tx_mask[it]:
                optimizer.zero_grad()
                if tx_objective == 'comm':
                    loss_tx = loss_c
                elif eta_r[it] == 1:
                    loss_tx = loss_r_pol
                elif eta_r[it] == 0:
                    loss_tx = loss_c
                else:
                    loss_tx = eta_r[it]*loss_r_pol + (1-eta_r[it])*loss_c
                loss_tx.backward(retain_graph=train_rx)
                optimizer.step()
                scheduler.step(loss_tx.item())
                if it % 2000 == 0:
                    print(f"The TX LRs on iteration {it} were {[optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))]}", flush=True)
            if train_rx:
                optimizer_rx.zero_grad()
                loss_rx = eta_r[it]*loss_r     #The communication loss does not depend on the RX
                loss_rx.backward()
                optimizer_rx.step()
                scheduler_rx.step(loss_rx.item())
                if it % 2000 == 0:
                    print(f"The RX LRs on iteration {it} were {[optimizer_rx.param_groups[i]['lr'] for i in range(len(optimizer_rx.param_groups))]}", flush=True)
        else:
            optimizer.zero_grad()
            if method == 'slcb' and eta_r[it] == 1:
                loss = loss_r
            elif method == 'slcb' and eta_r[it] == 0:
                loss = loss_c
            else:
                loss = eta_r[it]*loss_r + (1-eta_r[it])*loss_c
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())
            if it % 2000 == 0:
                print(f"The LRs on iteration {it} were {[optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))]}", flush=True)

        projectImpairments(network, device=device)

        loss_r_list.append(loss_r.item())
        loss_c_list.append(loss_c.item())

    scnr_dB = 10*np.log10(np.mean(scnr)) if len(scnr) > 0 else float('nan')
    return loss_r_list, loss_c_list, scnr_dB

def saveNetwork(save_path, network):
    '''Save the learned parameters Psi.'''
    torch.save({'model': network.state_dict()}, save_path)
    print('Model saved successfully', flush=True)


#######################################################################
# Testing functions                                                   #
#######################################################################
def getSensingMetrics(T_max, adm_max, T_true, delta, est_pos=None, true_pos=None,
                      gamma_gospa=None, mu_gospa=None, p_gospa=None):
    '''Detection probability (1 - p_md of eq. (36)), false-alarm probability (eq. (37)) and,
    if positions are given, mean GOSPA (eq. (38)) and mean localization error, for the OMP
    termination threshold delta. est_pos has shape (2, n_test, T_max) and true_pos is a list
    of (T, 2) tensors.'''
    device = adm_max.device
    n_test = len(T_true)

    mask = adm_max >= delta
    T_hat = torch.sum(mask, dim=1, keepdim=True)
    p_d = ( torch.minimum(T_hat, T_true).sum()*1.0/T_true.sum() ).item()
    p_fa =  ( (torch.maximum(T_hat, T_true)-T_true).sum()*1.0 \
            /(n_test*T_max - T_true.sum()) ).item()

    if (est_pos is None) or (true_pos is None):
        gospa_pos = None
        pos_err = None
    else:
        T_hat_list = T_hat.flatten().tolist()
        est_pos_mask = est_pos[:,mask].transpose(-1,-2).view(-1,2)
        est_pos_thr = est_pos_mask.split(T_hat_list)
        gospa_list, pos_err_list = map(list,zip(*[gospa(est_pos_thr[i], true_pos[i], gamma_gospa, mu_gospa, p_gospa, device=device) for i in range(n_test)]))
        gospa_pos = torch.mean(torch.stack(gospa_list)).item()
        pos_err = torch.nanmean(torch.stack(pos_err_list)).item()

    return p_d, p_fa, gospa_pos, pos_err

def obtainThresholdsFixedPfa(T_max, adm_max, T_true, p_fa_target, delta_p_fa, delta):
    '''Iteratively adjust three OMP termination thresholds delta until all of them yield
    p_fa within p_fa_target +- delta_p_fa. The input delta is modified in place.'''
    print(f'Initital thresholds were: {delta}', flush=True)
    device = adm_max.device

    with torch.no_grad():
        p_fa = np.zeros((1,len(delta)))
        while ((np.max(p_fa) > p_fa_target + delta_p_fa) or (np.min(p_fa) < p_fa_target - delta_p_fa)):
            p_fa = []
            for t in range(len(delta)):
                _, p_fa_temp, _, _ = getSensingMetrics(T_max, adm_max, T_true, delta[t])
                p_fa.append(p_fa_temp)

            if p_fa_target < np.min(p_fa):
                delta += torch.std(delta)
            elif p_fa_target > np.max(p_fa):
                delta -= torch.std(delta)
            else:
                if np.max(p_fa) > p_fa_target + delta_p_fa:
                    delta = torch.linspace((delta[0] + delta[1]) / 2.0, delta[2], 3, device=device)
                if np.min(p_fa) < p_fa_target - delta_p_fa:
                    delta = torch.linspace(delta[0], (delta[1] + delta[2]) / 2.0, 3, device=device)

    print(f'Final thresholds to yield Pfa={p_fa_target} are: {delta}', flush=True)
    return delta

def _polarToCartesian(est_theta, est_R, true_theta, true_R):
    '''(theta, R) estimates of shape (n_test, T_max) and lists of true values to (x, y).
    Returns the estimates as a (2, n_test, T_max) tensor and the true positions as a list
    of (T, 2) tensors.'''
    est_x = est_R * torch.cos(est_theta)
    est_y = est_R * torch.sin(est_theta)
    est_pos = torch.cat((est_x[None,:,:], est_y[None,:,:]), dim=0)
    true_pos = [torch.cat(((true_R[i]*torch.cos(true_theta[i])).view(-1,1),
                           (true_R[i]*torch.sin(true_theta[i])).view(-1,1)), dim=1)
                for i in range(len(true_theta))]
    return est_pos, true_pos

def _sensingRX(system, Y_s, x, theta_min, theta_max, R_min, R_max, xi_rx, omega_rx, ones_K):
    '''Sensing receiver at test time: remove x and run T_max OMP iterations over the sector
    with the dictionary of eq. (15) built from (xi_rx, omega_rx).'''
    B = Y_s.shape[0]
    Y_s /= ones_K @ x.permute(0,2,1)
    delta_angle = (theta_max-theta_min)/(system.N_theta-1)
    theta_grid_matrix = theta_min + delta_angle*torch.arange(system.N_theta, device=Y_s.device).reshape(1,-1)
    Phi_a, _ = steeringMatrix(theta_grid_matrix, theta_grid_matrix, omega_rx, system.lamb, xi_rx)
    Phi_a = Phi_a.transpose(-1,-2).reshape(B, system.N_theta, system.K).permute(0,2,1)
    return omp(Y_s, theta_min, theta_max, R_min, R_max, system.delta_f, system.T_max, Phi_a, system.N_theta, system.N_tau)

def computePositionsTesting(system, sens, A_tx, xi_rx, omega_rx, B, n_test, rng_gen=None, device='cuda'):
    '''Run n_test sensing realizations with the precoder f_s built from A_tx and the OMP
    receiver built from (xi_rx, omega_rx). Returns the ADM maxima (n_test, T_max), the true
    number of targets (n_test, 1), and the estimated and true positions (see _polarToCartesian).'''
    with torch.no_grad():
        T_max, S = system.T_max, system.S
        n_batches = n_test // B
        ones_K = torch.ones((system.K,1), device=device).type(torch.cfloat)

        T_true      = torch.empty(n_test, 1, dtype=torch.float32, device=device)
        true_theta  = []
        true_R      = []
        adm_max_all = torch.empty(n_test, T_max, dtype=torch.float32, device=device)
        est_theta   = torch.empty(n_test, T_max, dtype=torch.float32, device=device)
        est_R       = torch.empty(n_test, T_max, dtype=torch.float32, device=device)

        for i in range(n_batches):
            T = torch.randint(0,T_max+1,(B,), generator=rng_gen, device=device)
            msg = torch.randint(0,len(system.x_ref),size=(B*S,), dtype=torch.int64, generator=rng_gen, device=device)
            x = system.x_ref[msg].reshape(B, S, 1)

            theta_min, theta_max = generateInterval(sens.theta_mean_min, sens.theta_mean_max, sens.delta_theta_min, sens.delta_theta_max, B, rng_gen, device=device)
            R_min, R_max = generateInterval(sens.R_mean_min, sens.R_mean_max, sens.delta_R_min, sens.delta_R_max, B, rng_gen, device=device)
            f_s = createPrecoder(theta_min, theta_max, A_tx, system.N_theta, system.P, device=device)

            Y_s, theta, R, _ = sensingChannel(system, theta_min, theta_max, R_min, R_max, f_s, x, T, rng_gen)
            theta = theta.flatten().split(T.to(torch.int).tolist())
            R = R.flatten().split(T.to(torch.int).tolist())

            adm_max, theta_hat, R_hat, _ = _sensingRX(system, Y_s, x, theta_min, theta_max, R_min, R_max,
                                                      xi_rx, omega_rx, ones_K)

            T_true[i*B:(i+1)*B, 0] = T
            true_theta.extend(theta)
            true_R.extend(R)
            adm_max_all[i*B:(i+1)*B, :] = adm_max
            est_theta[i*B:(i+1)*B, :] = theta_hat
            est_R[i*B:(i+1)*B, :] = R_hat

        est_pos, true_pos = _polarToCartesian(est_theta, est_R, true_theta, true_R)

    return adm_max_all, T_true, est_pos, true_pos

def _collectSensingMetrics(T_max, adm_max, T_true, deltas, est_pos, true_pos, testing):
    '''getSensingMetrics for every threshold in deltas. Returns (p_d, p_fa, gospa, pos_err) as lists.'''
    p_d, p_fa, gospa_pos, pos_err = [], [], [], []
    for t in range(len(deltas)):
        p_d_temp, p_fa_temp, gospa_temp, pos_err_temp = getSensingMetrics(
            T_max, adm_max, T_true, deltas[t],
            est_pos, true_pos, testing.gamma_gospa, testing.mu_gospa, testing.p_gospa)
        p_d.append(p_d_temp)
        p_fa.append(p_fa_temp)
        gospa_pos.append(gospa_temp)
        pos_err.append(pos_err_temp)
    return p_d, p_fa, gospa_pos, pos_err

def _interpolatePfaMetrics(p_d, p_fa, gospa_pos, pos_err, target_log_p_fa=-2):
    '''Linearly interpolate (p_d, gospa, pos_err) at log10(p_fa) = target_log_p_fa; the returned
    p_fa is the mean of the collected values.'''
    p_fa_ordered = np.sort(p_fa)
    idx_order = np.argsort(p_fa)
    p_d_i     = np.interp(target_log_p_fa, np.log10(p_fa_ordered), np.array(p_d)[idx_order])
    p_fa_i    = np.array(p_fa).mean()
    gospa_i   = np.interp(target_log_p_fa, np.log10(p_fa_ordered), np.array(gospa_pos)[idx_order])
    pos_err_i = np.interp(target_log_p_fa, np.log10(p_fa_ordered), np.array(pos_err)[idx_order])
    return p_d_i, p_fa_i, gospa_i, pos_err_i

def testSensingROC(system, sens, A_tx, xi_rx, omega_rx, deltas, testing, B, n_test, rng_gen=None, device='cuda'):
    '''Sensing ROC over the thresholds deltas. Returns (p_d, p_fa, gospa, pos_err) as lists.'''
    print('**STARTED TESTING TO OBTAIN ROC CURVES**')
    with torch.no_grad():
        adm_max, T_true, est_pos, true_pos = computePositionsTesting(
            system, sens, A_tx, xi_rx, omega_rx, B, n_test, rng_gen, device=device)
        return _collectSensingMetrics(system.T_max, adm_max, T_true, deltas, est_pos, true_pos, testing)

def testSensingFixedPfa(system, sens, A_tx, xi_rx, omega_rx, testing, B, n_test, rng_gen=None, device='cuda'):
    '''Sensing metrics interpolated at p_fa = testing.p_fa_target. Returns (p_d, p_fa, gospa, pos_err).'''
    print('**STARTED TESTING WITH FIXED PFA**')
    with torch.no_grad():
        adm_max, T_true, est_pos, true_pos = computePositionsTesting(
            system, sens, A_tx, xi_rx, omega_rx, B, n_test, rng_gen, device=device)
        deltas = obtainThresholdsFixedPfa(system.T_max, adm_max, T_true, testing.p_fa_target,
                                          testing.delta_p_fa, torch.clone(testing.delta_init))
        metrics = _collectSensingMetrics(system.T_max, adm_max, T_true, deltas, est_pos, true_pos, testing)
        return _interpolatePfaMetrics(*metrics)

def testCommunication(system, comm, A_tx, B, n_test, rng_gen=None, device='cuda'):
    '''Communication SNR [dB] and SER (eq. (39)) with the precoder f_c built from A_tx.'''
    S = system.S
    n_batches = n_test // B

    num_errors = torch.tensor(0,dtype=torch.float32, device=device)
    mean_abs_kappa_sq = torch.zeros(S,1, device=device)

    with torch.no_grad():
        for b in range(n_batches):
            T_c = torch.randint(1,system.T_max_c+1,(B,), generator=rng_gen, device=device)
            theta_min_c, theta_max_c = generateInterval(comm.theta_mean_min, comm.theta_mean_max, comm.delta_theta_min, comm.delta_theta_max, B, rng_gen, device=device)
            R_min_c, R_max_c = generateInterval(comm.R_mean_min, comm.R_mean_max, comm.delta_R_min, comm.delta_R_max, B, rng_gen, device=device)
            f_c = createPrecoder(theta_min_c, theta_max_c, A_tx, system.N_theta, system.P, device=device)
            msg = torch.randint(0,len(system.x_ref),size=(B*S,), dtype=torch.int64, generator=rng_gen, device=device)
            x = system.x_ref[msg].reshape(B, S, 1)

            y_c, kappa = commChannel(system, theta_min_c, theta_max_c, R_min_c, R_max_c, f_c, x, T_c, rng_gen, device=device)
            x_hat = MLdecoder(y_c, kappa, system.x_ref)

            num_errors += (msg != x_hat.flatten()).sum()
            mean_abs_kappa_sq += torch.mean(torch.abs(kappa)**2, dim=0)

            if (b+1)%2000 == 0:
                print(f'Iteration {b+1} out of {n_batches}', flush=True)

        mean_abs_kappa_sq /= n_batches
        snr_c = torch.mean(10*torch.log10(mean_abs_kappa_sq/system.noise_var))
        ser = num_errors/(n_test * S)

    return snr_c, ser

def testISAC(system, sens, comm, A_tx, xi_rx, omega_rx, testing, omega_r_try, B, n_test, rng_gen=None, device='cuda'):
    '''ISAC trade-off: for each omega_r in omega_r_try, joint sensing and communication over
    the same realizations. Returns the arrays (p_d, p_fa, gospa, ser, snr_c), with the
    sensing metrics interpolated at p_fa = testing.p_fa_target.'''
    print('**STARTED TESTING WITH FIXED PFA**')
    with torch.no_grad():
        T_max, S, N_theta, P = system.T_max, system.S, system.N_theta, system.P
        p_d_isac = np.empty(len(omega_r_try))
        p_fa_isac = np.empty(len(omega_r_try))
        gospa_isac = np.empty(len(omega_r_try))
        ser_isac = np.empty(len(omega_r_try))
        snr_c_isac = np.empty(len(omega_r_try))
        phi_zero = torch.tensor(0, device=device)
        ones_K = torch.ones((system.K,1), device=device).type(torch.cfloat)
        for k, omega_r in enumerate(omega_r_try):
            num_errors = torch.tensor(0,dtype=torch.float32, device=device)
            mean_abs_kappa_sq = torch.zeros(S,1, device=device)
            n_batches = n_test // B

            T_true      = torch.empty(n_test, 1, dtype=torch.float32, device=device)
            true_theta  = []
            true_R      = []
            adm_max_all = torch.empty(n_test, T_max, dtype=torch.float32, device=device)
            est_theta   = torch.empty(n_test, T_max, dtype=torch.float32, device=device)
            est_R       = torch.empty(n_test, T_max, dtype=torch.float32, device=device)

            for i in range(n_batches):
                T = torch.randint(0,T_max+1,(B,), generator=rng_gen, device=device)
                T_c = torch.randint(1,system.T_max_c+1,(B,), generator=rng_gen, device=device)
                msg = torch.randint(0,len(system.x_ref),size=(B*S,), dtype=torch.int64, generator=rng_gen, device=device)
                x = system.x_ref[msg].reshape(B, S, 1)

                theta_min_s, theta_max_s = generateInterval(sens.theta_mean_min, sens.theta_mean_max, sens.delta_theta_min, sens.delta_theta_max, B, rng_gen, device=device)
                theta_min_c, theta_max_c = generateInterval(comm.theta_mean_min, comm.theta_mean_max, comm.delta_theta_min, comm.delta_theta_max, B, rng_gen, device=device)
                R_min_s, R_max_s = generateInterval(sens.R_mean_min, sens.R_mean_max, sens.delta_R_min, sens.delta_R_max, B, rng_gen, device=device)
                R_min_c, R_max_c = generateInterval(comm.R_mean_min, comm.R_mean_max, comm.delta_R_min, comm.delta_R_max, B, rng_gen, device=device)
                f_s = createPrecoder(theta_min_s, theta_max_s, A_tx, N_theta, P, device=device)
                f_c = createPrecoder(theta_min_c, theta_max_c, A_tx, N_theta, P, device=device)
                f_tx = isacPrecoder(f_s, f_c, omega_r, phi_zero, P)

                Y_s, theta, R, _ = sensingChannel(system, theta_min_s, theta_max_s, R_min_s, R_max_s, f_tx, x, T, rng_gen)
                theta = theta.flatten().split(T.to(torch.int).tolist())
                R = R.flatten().split(T.to(torch.int).tolist())
                y_c, kappa = commChannel(system, theta_min_c, theta_max_c, R_min_c, R_max_c, f_tx, x, T_c, rng_gen, device=device)

                adm_max, theta_hat, R_hat, _ = _sensingRX(system, Y_s, x, theta_min_s, theta_max_s, R_min_s, R_max_s,
                                                          xi_rx, omega_rx, ones_K)

                T_true[i*B:(i+1)*B, 0] = T
                true_theta.extend(theta)
                true_R.extend(R)
                adm_max_all[i*B:(i+1)*B, :] = adm_max
                est_theta[i*B:(i+1)*B, :] = theta_hat
                est_R[i*B:(i+1)*B, :] = R_hat

                x_hat = MLdecoder(y_c, kappa, system.x_ref)
                num_errors += (msg != x_hat.flatten()).sum()
                mean_abs_kappa_sq += torch.mean(torch.abs(kappa)**2, dim=0)

            est_pos, true_pos = _polarToCartesian(est_theta, est_R, true_theta, true_R)
            deltas = obtainThresholdsFixedPfa(T_max, adm_max_all, T_true, testing.p_fa_target,
                                              testing.delta_p_fa, torch.clone(testing.delta_init))
            metrics = _collectSensingMetrics(T_max, adm_max_all, T_true, deltas, est_pos, true_pos, testing)
            p_d_i, p_fa_i, gospa_i, _ = _interpolatePfaMetrics(*metrics)

            mean_abs_kappa_sq /= n_batches
            snr_c = torch.mean(10*torch.log10(mean_abs_kappa_sq/system.noise_var))
            ser = num_errors/(n_test * S)

            p_d_isac[k] = p_d_i
            p_fa_isac[k] = p_fa_i
            gospa_isac[k] = gospa_i
            ser_isac[k] = ser.cpu().detach().numpy()
            snr_c_isac[k] = snr_c.cpu().detach().numpy()

    return p_d_isac, p_fa_isac, gospa_isac, ser_isac, snr_c_isac


#######################################################################
# Experiment helpers                                                  #
#######################################################################
#File names, evaluation phases and saved results shared by the scripts in methods/.

def fileName(method, eta_r, P, train_it, impaired, ref_sensor, seed, sigma=None, sens_loss=None, I_omp=None):
    '''Name of the results/model files of a trained method.'''
    eta_tag = str(np.round(eta_r[0].item(), 2)) if torch.all(eta_r == eta_r[0]) else 'rand'
    name = method + '_eta_' + eta_tag + '_P_' + '{:.0e}'.format(P) + '_its_' + str(train_it)
    if sigma is not None:
        name += '_sigma_' + '{:.3g}'.format(float(sigma))
    if sens_loss is not None:
        name += '_loss_' + sens_loss + ('_I_' + str(I_omp) if sens_loss == 'residual' else '')
    return name + commonTag(P, impaired, ref_sensor, seed, with_P=False)

def commonTag(P, impaired, ref_sensor, seed, with_P=True):
    '''File-name suffix with the transmit power, impairment knowledge, reference sensor and seed.'''
    tag = '_P_' + '{:.0e}'.format(P) if with_P else ''
    tag += '_imp' if impaired else '_known'
    tag += '_ref' if ref_sensor else ''
    return tag + '_seed_' + str(seed)

def loadNetwork(load_path, network):
    '''Load the learned parameters Psi into network (in place).'''
    checkpoint = torch.load(load_path, map_location=network.xi_tx.device)
    network.load_state_dict(checkpoint['model'])

def txDictionary(xi_tx, omega_tx, theta_grid, lamb):
    '''(K x N_theta) TX steering vectors over theta_grid for the precoder of eq. (13).'''
    A_tx, _ = steeringMatrix(theta_grid, theta_grid, omega_tx, lamb, xi_tx)
    return A_tx

def runSensingROC(system, sens, A_tx, xi_rx, omega_rx, deltas, testing, B, n_test, rng_gen, device='cuda'):
    '''Sensing ROC with rng_gen reset to its testing seed. Returns (p_d, p_fa, gospa, pos_err).'''
    rng_gen.manual_seed(1998)
    return testSensingROC(system, sens, A_tx, xi_rx, omega_rx, deltas, testing, B, n_test,
                          rng_gen=rng_gen, device=device)

def runCommSweep(system, comm, A_tx, R_max_c_sweep, R_min_c, B, n_test, rng_gen, device='cuda'):
    '''Communication SNR and SER for every maximum UE range in R_max_c_sweep.'''
    snr_c, ser = [], []
    for k, R_max_c in enumerate(R_max_c_sweep):
        print(f'Testing range {k} out of {len(R_max_c_sweep)}')
        R_mean_c = (R_min_c+R_max_c)/2.0
        delta_R_c = R_max_c-R_min_c
        comm_k = replace(comm, R_mean_min=R_mean_c, R_mean_max=R_mean_c, delta_R_min=delta_R_c, delta_R_max=delta_R_c)
        snr_c_k, ser_k = testCommunication(system, comm_k, A_tx, B, n_test, rng_gen=rng_gen, device=device)
        snr_c.append(snr_c_k.item())
        ser.append(ser_k.item())
    return snr_c, ser

def runISAC(system, sens, comm, A_tx, xi_rx, omega_rx, testing, B, n_test, device='cuda'):
    '''ISAC trade-off over omega_r (global RNG). Returns (p_d, p_fa, gospa, ser, snr_c).'''
    omega_r_try = torch.cat((torch.logspace(-2,-1,5, device=device),torch.linspace(0.2,1,5, device=device)))
    return testISAC(system, sens, comm, A_tx, xi_rx, omega_rx, testing, omega_r_try, B, n_test,
                    rng_gen=None, device=device)

def runSnrSweep(system, sens, A_tx, xi_rx, omega_rx, testing, P_list, B, n_test, rng_gen, device='cuda'):
    '''Fixed-p_fa sensing metrics for every transmit power in P_list.'''
    p_d, p_fa, gospa_pos, pos_err = [], [], [], []
    for P_try in P_list:
        p_d_k, p_fa_k, gospa_k, pos_err_k = testSensingFixedPfa(
            replace(system, P=P_try), sens, A_tx, xi_rx, omega_rx, testing, B, n_test, rng_gen=rng_gen, device=device)
        p_d.append(p_d_k)
        p_fa.append(p_fa_k)
        gospa_pos.append(gospa_k)
        pos_err.append(pos_err_k)
    return {'p_d_snr': p_d, 'p_fa_snr': p_fa, 'gospa_snr': gospa_pos, 'pos_err_snr': pos_err, 'P_list': P_list}

def evaluateNetwork(network, system, sens, comm, comm_isac, testing, test_params, R_max_c_sweep, R_min_c,
                    B, n_test, n_test_comm, rng_test, device='cuda'):
    '''Evaluate learned parameters: sensing ROC, communication SNR sweep and ISAC trade-off
    (with initial fixed-p_fa thresholds testing.delta_init). Returns the results to save.'''
    network.eval()
    A_tx = txDictionary(network.xi_tx, network.omega_tx, system.theta_grid, system.lamb)
    deltas = torch.logspace(np.log10(2e-4), np.log10(2.2e-3), 10, device=device)
    p_d, p_fa, gospa_pos, pos_err = runSensingROC(system, sens, A_tx, network.xi_rx, network.omega_rx,
                                                  deltas, test_params, B, n_test, rng_test, device=device)
    snr_c, ser = runCommSweep(system, comm, A_tx, R_max_c_sweep, R_min_c, B, n_test_comm, rng_test, device=device)
    p_d_isac, p_fa_isac, gospa_isac, ser_isac, snr_c_isac = runISAC(system, sens, comm_isac, A_tx,
                                                                    network.xi_rx, network.omega_rx, testing,
                                                                    B, n_test, device=device)
    return {'p_d_roc': p_d, 'p_fa_roc': p_fa, 'gospa_roc': gospa_pos, 'pos_err_roc': pos_err,
            'snr_c_sweep': snr_c, 'ser_sweep': ser,
            'p_d_isac': p_d_isac, 'p_fa_isac': p_fa_isac, 'gospa_isac': gospa_isac,
            'ser_isac': ser_isac, 'snr_c_isac': snr_c_isac}

def impairmentsToSave(network, system):
    '''True impairments Xi and learned parameters Psi as numpy arrays.'''
    return {'gamma_tx': system.gamma_tx.cpu().detach().numpy(), 'gamma_rx': system.gamma_rx.cpu().detach().numpy(),
            'p_tx': system.p_tx.cpu().detach().numpy(), 'p_rx': system.p_rx.cpu().detach().numpy(),
            'xi_tx': network.xi_tx.cpu().detach().numpy(), 'xi_rx': network.xi_rx.cpu().detach().numpy(),
            'omega_tx': network.omega_tx.cpu().detach().numpy(), 'omega_rx': network.omega_rx.cpu().detach().numpy()}
