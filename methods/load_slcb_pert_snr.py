# -*- coding: utf-8 -*-
'''
Load a model trained with slcb_pert.py (same arguments, including -s, and parameters) and test the
sensing performance for different transmit powers (Fig. 12).
'''
from ..lib.simulation_parameters import *

omega_r = torch.rand(train_it, device=device)
eta_r = torch.clone(omega_r)
file_name = fileName('slcb_pert', eta_r, P, train_it, impaired, ref_sensor, seed, sigma=sigma)
print(file_name, flush=True)

loadNetwork('models/' + file_name + '_model', network)
network.eval()
A_tx = txDictionary(network.xi_tx, network.omega_tx, theta_grid, lamb)
testing = replace(test_params, delta_init=torch.linspace(1.15e-2, 1.2e-2, 3, device=device))
results = runSnrSweep(system, sens, A_tx, network.xi_rx, network.omega_rx, testing, torch.logspace(-7,1,9),
                      B, n_test, rng_test, device=device)

np.savez('results/load_snr_' + file_name, **results)
