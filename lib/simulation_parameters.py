# -*- coding: utf-8 -*-
'''
Simulation parameters (Table II), true impairments Xi, learnable parameters Psi and optimizers.
Imported (and executed) by every script in methods/.
'''
import torch
import numpy as np
import argparse

from ..lib.functions import *
from ..lib.learnable_parameters import *

######## Command-line arguments ########
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--impaired", type=int, default=1,
                    help="1: no impairment knowledge, Psi is initialized to the ideal ULA. 0: Psi is initialized to the true impairments Xi.")
parser.add_argument("-s", "--sigmaIdx", type=int, default=0,
                    help="Index of the precoder perturbation std sigma in sigma_list (3 -> sigma = 0.05, as in the paper).")
parser.add_argument("-n", "--seed", type=int, default=10,
                    help="Seed of the impairment realization.")
parser.add_argument("--sensLoss", type=str, default='adm', choices=['adm', 'residual'],
                    help="GF UL sensing loss: 'adm' (maximum ADM response, eq. (23)) or 'residual' (OMP residual norm, eq. (24)).")
parser.add_argument("--ompIt", type=int, default=1,
                    help="Number of OMP iterations I of the residual loss (eq. (24)).")
parser.add_argument("--refSensor", type=int, default=0,
                    help="1: the first element of each ULA is a known reference sensor that is not learned (eq. (35)). 0: all elements are learned.")
parser.add_argument("--etaFixed", type=float, default=None,
                    help="GF UL only: fixed eta_r for every iteration (0.0: only communication, 1.0: only sensing). If not given, eta_r = omega_r ~ U[0, 1].")
parser.add_argument("--clutterVar", type=float, default=None,
                    help="GF UL only: clutter gain variance sigma_c2 of the training sensing channel (up to C_max = 5 scatterers). If not given, training is clutter-free.")
parser.add_argument("--knownTx", type=int, default=0,
                    help="GF UL only: 1: the TX impairments are known (Psi_tx = Xi_tx) and only Psi_rx is learned (Fig. 4). 0: Psi_tx is also learned.")
args = parser.parse_args()

#Reproducibility
torch.manual_seed(10)
np.random.seed(10)
torch.cuda.manual_seed(10)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
#Required by torch.use_deterministic_algorithms on CUDA: https://docs.nvidia.com/cuda/cublas/#results-reproducibility
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#Impairments depend on the seed; training and testing data are the same for every impairment realization
seed       = args.seed
rng_imp    = torch.Generator(device=device)
rng_train  = torch.Generator(device=device)
rng_test   = torch.Generator(device=device)
rng_imp.manual_seed(seed)
rng_train.manual_seed(10)
rng_test.manual_seed(1998)

######## System parameters ########
impaired    = args.impaired
T_max       = 5                                                                     #Maximum number of targets
T_max_c     = 6                                                                     #Maximum number of BS-UE paths (including LoS)
P           = torch.tensor(1e-1, dtype=torch.float32, device=device)                #Transmit power [W]
sigma_mean  = torch.tensor(1.0, dtype=torch.float32, device=device)                 #Mean RCS [m^2]
K           = 64                                                                    #Antennas per ULA
S           = 256                                                                   #Subcarriers
delta_f     = torch.tensor(240e3, dtype=torch.float32, device=device)               #Subcarrier spacing [Hz]
fc          = torch.tensor(60e9, dtype=torch.float32, device=device)                #Carrier frequency [Hz]
lamb        = 3e8 / fc                                                              #Wavelength [m]
boltzmann   = torch.tensor(1.38e-23, device=device)                                 #Boltzmann constant [J/K]
sys_temp    = torch.tensor(290, device=device)                                      #System temperature [K]
N0          = 10*torch.log10(boltzmann*sys_temp/1e-3)                               #Noise PSD [dBm/Hz]
noise_fig   = torch.tensor(8, device=device)                                        #Noise figure [dB]
noise_var   = 10**((N0+noise_fig)/10)*1e-3*S*delta_f                               #Noise variance N0*S*delta_f [W]
x_ref       = MPSK(4, np.pi/4, device=device)                                       #QPSK constellation
T_cp        = 0.07/delta_f                                                          #Cyclic prefix [s]
R_cp        = T_cp*3e8                                                              #Range equivalent to T_cp [m]
#Sensing range uncertainty region [R_min, R_max] and field of view
R_min       = torch.tensor(10, dtype=torch.float32, device=device)
R_max       = torch.tensor(43.75, dtype=torch.float32, device=device)
theta_fov   = torch.tensor(np.pi/2, dtype=torch.float32, device=device)
#Grids of the precoder (N_theta over the FoV) and of the OMP (N_theta x N_tau over the sector)
N_theta     = 720
N_tau       = 200
theta_grid  = torch.linspace(-theta_fov, theta_fov, N_theta, device=device)

######## True impairments Xi (Sec. IV-A) ########
p_first     = -((K-1)/2) * (lamb/2)                                                 #First element of the ideal ULA (centered around 0)
eps_max     = lamb/5                                                                #ADIs: [eps_p]_k ~ U[-lambda/5, lambda/5]
p_tx        = generatePositionImpairments(eps_max, K, lamb, p_first, rng_imp, device=device)
p_rx        = generatePositionImpairments(eps_max, K, lamb, p_first, rng_imp, device=device)
min_gain_gp = 0.95; max_gain_gp = 1.00
min_phase_gp = -np.pi/2.0; max_phase_gp = np.pi/2.0
gamma_tx    = generateGPImpairments(min_gain_gp, max_gain_gp, min_phase_gp, max_phase_gp, K, rng_imp, device=device)
gamma_rx    = generateGPImpairments(min_gain_gp, max_gain_gp, min_phase_gp, max_phase_gp, K, rng_imp, device=device)
torch.set_printoptions(precision=10)
print('TX positions\n\t', p_tx.flatten(), flush=True)
print('RX positions\n\t', p_rx.flatten(), flush=True)
print('TX GPIs\n\t', gamma_tx.flatten(), flush=True)
print('RX GPIs\n\t', gamma_rx.flatten(), flush=True)
torch.set_printoptions(precision=5)

######## Uncertainty regions (Table II) ########
#Sensing (training and testing) and communication angles: theta_mean ~ U[-60, 60] deg, Delta_theta ~ U[10, 20] deg
theta_mean_min  = torch.tensor(-60*np.pi/180, dtype=torch.float32, device=device)
theta_mean_max  = torch.tensor(60*np.pi/180, dtype=torch.float32, device=device)
delta_theta_min = torch.tensor(10*np.pi/180, dtype=torch.float32, device=device)
delta_theta_max = torch.tensor(20*np.pi/180, dtype=torch.float32, device=device)
#Sensing ranges: fixed to [R_min, R_max]
R_mean_s        = (R_min + R_max)/2.0
delta_R_s       = R_max - R_min
#UE ranges: [R_min_c, R_max_c], and [R_min_c, R_max_c_isac] for the ISAC test
R_min_c         = torch.tensor(10.0, dtype=torch.float32, device=device)
R_max_c         = torch.tensor(200.0, dtype=torch.float32, device=device)
R_max_c_isac    = torch.tensor(75.0, dtype=torch.float32, device=device)
#Maximum UE ranges of the communication SNR sweep
R_max_c_sweep   = torch.tensor([20, 30, 50, 75, 115, 175, 270, 420, 650, 1000], dtype=torch.float32, device=device)

sens      = SectorParams(theta_mean_min, theta_mean_max, delta_theta_min, delta_theta_max,
                         R_mean_s, R_mean_s, delta_R_s, delta_R_s)
comm      = SectorParams(theta_mean_min, theta_mean_max, delta_theta_min, delta_theta_max,
                         (R_min_c + R_max_c)/2.0, (R_min_c + R_max_c)/2.0,
                         R_max_c - R_min_c, R_max_c - R_min_c)
comm_isac = replace(comm, R_mean_min=(R_min_c + R_max_c_isac)/2.0, R_mean_max=(R_min_c + R_max_c_isac)/2.0,
                    delta_R_min=R_max_c_isac - R_min_c, delta_R_max=R_max_c_isac - R_min_c)

######## Testing parameters ########
#Each training iteration processes T_max minibatches of B samples
B           = 800
n_test      = (int(1.1e5) // B)*B                                                   #Sensing and ISAC testing samples
n_test_comm = (int(1.1e6) // B)*B                                                   #Communication testing samples
num_thr     = 10                                                                    #Thresholds of the ROC curves
p_fa_target = 1e-2
delta_p_fa  = 1e-3
gamma_gospa = R_mean_s-R_mean_s+delta_R_s                                           #GOSPA cut-off R_max - R_min
mu_gospa    = torch.tensor(2, dtype=torch.float32, device=device)
p_gospa     = torch.tensor(2, dtype=torch.float32, device=device)
test_params = TestParams(p_fa_target=p_fa_target, delta_p_fa=delta_p_fa,
                         delta_init=torch.linspace(9e-4, 1.1e-3, 3, device=device),
                         mu_gospa=mu_gospa, p_gospa=p_gospa, gamma_gospa=gamma_gospa)

system = SystemParams(T_max=T_max, T_max_c=T_max_c, P=P, sigma_mean=sigma_mean, R_cp=R_cp,
                      K=K, S=S, delta_f=delta_f, lamb=lamb, N_theta=N_theta, N_tau=N_tau,
                      theta_grid=theta_grid, x_ref=x_ref, noise_var=noise_var,
                      gamma_tx=gamma_tx, gamma_rx=gamma_rx, p_tx=p_tx, p_rx=p_rx)

######## Learnable parameters Psi ########
#Psi(0): ideal ULA without GPIs (no impairment knowledge) or the true impairments
if impaired:
    p_ideal_spacing = torch.cat((torch.tensor([p_first], device=device), lamb/2 * torch.ones((K-1,), device=device)))
    omega_tx_init = torch.cumsum(p_ideal_spacing, dim=0)
    omega_rx_init = torch.clone(omega_tx_init)
    xi_tx_init    = torch.ones((K,1), dtype=torch.cfloat, device=device)
    xi_rx_init    = torch.ones((K,1), dtype=torch.cfloat, device=device)
else:
    omega_tx_init = torch.clone(p_tx); omega_rx_init = torch.clone(p_rx)
    xi_tx_init = torch.clone(gamma_tx); xi_rx_init = torch.clone(gamma_rx)
#Reference sensor (eq. (35)): the first element of each ULA is known, also by the baselines
ref_sensor = bool(args.refSensor)
ref_idx    = 0
if ref_sensor:
    omega_tx_init[ref_idx] = p_tx[ref_idx]
    omega_rx_init[ref_idx] = p_rx[ref_idx]
    xi_tx_init[ref_idx]    = gamma_tx[ref_idx]
    xi_rx_init[ref_idx]    = gamma_rx[ref_idx]
    print(f'Antenna element {ref_idx} is assumed perfectly known (it is not calibrated)', flush=True)

network = LearnableImpairments(xi_tx_init.cpu(), xi_rx_init.cpu(),
                               omega_tx_init.cpu(), omega_rx_init.cpu()).to(device)
if ref_sensor:
    setReferenceSensor(network, ref_idx, p_tx, p_rx, gamma_tx, gamma_rx)

######## Optimization (Sec. IV-A) ########
train_it  = 10000
lr_xi     = 1e-2
lr_omega  = 1e-4
#Precoder perturbation std sigma
sigma_list = [lamb.cpu()*5, lamb.cpu()*1e-2, lamb.cpu()*2, lamb.cpu()*10, lamb.cpu()*50, lamb.cpu()*100, lamb.cpu()*199]
sigma      = sigma_list[args.sigmaIdx]
#SLCB: one optimizer for Psi. GF UL: one optimizer per side (the other side has a zero learning rate)
opt_slcb = torch.optim.Adam([
    {'params': [network.xi_tx], 'lr': lr_xi},
    {'params': [network.xi_rx], 'lr': lr_xi},
    {'params': [network.omega_tx], 'lr': lr_omega},
    {'params': [network.omega_rx], 'lr': lr_omega}
])
opt_tx = torch.optim.Adam([
    {'params': [network.xi_tx], 'lr': lr_xi},
    {'params': [network.xi_rx], 'lr': 0.0},
    {'params': [network.omega_tx], 'lr': lr_omega},
    {'params': [network.omega_rx], 'lr': 0.0}
])
opt_rx = torch.optim.Adam([
    {'params': [network.xi_tx], 'lr': 0.0},
    {'params': [network.xi_rx], 'lr': lr_xi},
    {'params': [network.omega_tx], 'lr': 0.0},
    {'params': [network.omega_rx], 'lr': lr_omega}
])
sch_slcb = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_slcb, factor=0.5, patience=1000, cooldown=500)
sch_tx   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_tx, factor=0.5, patience=1000, cooldown=500)
sch_rx   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_rx, factor=0.5, patience=1000, cooldown=500)
