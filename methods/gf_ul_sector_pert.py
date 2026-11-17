# -*- coding: utf-8 -*-
'''
GF UL calibration with eta_r = omega_r ~ U[0, 1] and an imperfect angular prior (Fig. 10): the
TX precoder and the RX OMP grid use the sector endpoints plus N(0, sigma_theta^2), while the
targets and the UE lie in the true sector. Testing uses the true sectors.
'''
from ..lib.simulation_parameters import *

sigma_theta = 5*np.pi/180

omega_r = torch.rand(train_it, device=device)
eta_r = torch.clone(omega_r)

file_name = fileName('gf_ul_sector_pert_' + str(np.round(sigma_theta*180/np.pi,1)) + 'deg', eta_r, P, train_it,
                     impaired, ref_sensor, seed, sigma=sigma, sens_loss=args.sensLoss, I_omp=args.ompIt)
print(file_name, flush=True)

### Training ###
loss_r, loss_c, scnr_s = train(network, 'gf_ul', system, sens, comm, B, train_it, omega_r, eta_r,
                               opt_tx, sch_tx, optimizer_rx=opt_rx, scheduler_rx=sch_rx, sigma=sigma,
                               sens_loss=args.sensLoss, I_omp=args.ompIt, sigma_theta=sigma_theta,
                               rng_gen=rng_train, device=device)
saveNetwork('models/' + file_name + '_model', network)

### Testing ###
testing = replace(test_params, delta_init=torch.linspace(1.15e-2, 1.2e-2, 3, device=device))
results = evaluateNetwork(network, system, sens, comm, comm_isac, testing, test_params, R_max_c_sweep, R_min_c,
                          B, n_test, n_test_comm, rng_test, device=device)

np.savez('results/' + file_name,
         omega_r=omega_r.cpu().detach().numpy(), eta_r=eta_r.cpu().detach().numpy(), sigma_theta=sigma_theta,
         loss_r=loss_r, loss_c=loss_c, scnr_s=scnr_s,
         **results, **impairmentsToSave(network, system))
