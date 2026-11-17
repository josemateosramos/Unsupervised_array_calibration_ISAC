# -*- coding: utf-8 -*-
'''
Load a model trained with gf_ul.py (same arguments and parameters) and test the sensing
performance for different transmit powers (Fig. 12).
'''
from ..lib.simulation_parameters import *

omega_r = torch.rand(train_it, device=device)
eta_r = torch.clone(omega_r)
file_name = fileName('gf_ul', eta_r, P, train_it, impaired, ref_sensor, seed,
                     sigma=sigma, sens_loss=args.sensLoss, I_omp=args.ompIt)
print(file_name, flush=True)

loadNetwork('models/' + file_name + '_model', network)
network.eval()
A_tx = txDictionary(network.xi_tx, network.omega_tx, theta_grid, lamb)
testing = replace(test_params, delta_init=torch.linspace(3e-4, 5e-3, 3, device=device))
results = runSnrSweep(system, sens, A_tx, network.xi_rx, network.omega_rx, testing, torch.logspace(-7,1,9),
                      B, n_test, rng_test, device=device)

np.savez('results/load_snr_' + file_name, **results)
