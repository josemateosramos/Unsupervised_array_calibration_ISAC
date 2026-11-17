# -*- coding: utf-8 -*-
'''
GF UL calibration with eta_r = omega_r ~ U[0, 1] and a modified training angular sector
(Fig. 11): theta_mean ~ U[-phi, phi] and Delta_theta ~ U[delta_theta_deg] for the targets and
the UE. Testing uses the sectors of Table II.
'''
from ..lib.simulation_parameters import *

phi_deg = 10.0
delta_theta_deg = (5.0, 5.0)

def trainSector(base_sector, phi_deg, delta_theta_deg, device):
    '''Copy of base_sector with theta_mean ~ U[-phi_deg, phi_deg] and Delta_theta ~ U[delta_theta_deg].'''
    return replace(base_sector,
                   theta_mean_min=torch.tensor(-phi_deg*np.pi/180, dtype=torch.float32, device=device),
                   theta_mean_max=torch.tensor(phi_deg*np.pi/180, dtype=torch.float32, device=device),
                   delta_theta_min=torch.tensor(delta_theta_deg[0]*np.pi/180, dtype=torch.float32, device=device),
                   delta_theta_max=torch.tensor(delta_theta_deg[1]*np.pi/180, dtype=torch.float32, device=device))

sens_train = trainSector(sens, phi_deg, delta_theta_deg, device)
comm_train = trainSector(comm, phi_deg, delta_theta_deg, device)

omega_r = torch.rand(train_it, device=device)
eta_r = torch.clone(omega_r)

sector_tag = '_phi_' + str(phi_deg) + '_dtheta_' + str(delta_theta_deg[0]) + '_' + str(delta_theta_deg[1])
file_name = fileName('gf_ul_train_sector' + sector_tag, eta_r, P, train_it, impaired, ref_sensor, seed,
                     sigma=sigma, sens_loss=args.sensLoss, I_omp=args.ompIt)
print(file_name, flush=True)

### Training ###
loss_r, loss_c, scnr_s = train(network, 'gf_ul', system, sens_train, comm_train, B, train_it, omega_r, eta_r,
                               opt_tx, sch_tx, optimizer_rx=opt_rx, scheduler_rx=sch_rx, sigma=sigma,
                               sens_loss=args.sensLoss, I_omp=args.ompIt, rng_gen=rng_train, device=device)
saveNetwork('models/' + file_name + '_model', network)

### Testing ###
testing = replace(test_params, delta_init=torch.linspace(1.15e-2, 1.2e-2, 3, device=device))
results = evaluateNetwork(network, system, sens, comm, comm_isac, testing, test_params, R_max_c_sweep, R_min_c,
                          B, n_test, n_test_comm, rng_test, device=device)

np.savez('results/' + file_name,
         omega_r=omega_r.cpu().detach().numpy(), eta_r=eta_r.cpu().detach().numpy(),
         phi_deg=phi_deg, delta_theta_deg=np.array(delta_theta_deg, dtype=np.float32),
         loss_r=loss_r, loss_c=loss_c, scnr_s=scnr_s,
         **results, **impairmentsToSave(network, system))
