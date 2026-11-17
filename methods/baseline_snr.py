# -*- coding: utf-8 -*-
'''
Model-based sensing baseline for different transmit powers (Fig. 12): fixed-p_fa metrics with
the impairments assumed by the BS.
'''
from ..lib.simulation_parameters import *

file_name = 'baseline_snr' + commonTag(P, impaired, ref_sensor, seed)
print(file_name, flush=True)

A_tx = txDictionary(xi_tx_init, omega_tx_init, theta_grid, lamb)
P_list = torch.logspace(-7,1,9)
results = runSnrSweep(system, sens, A_tx, xi_rx_init, omega_rx_init, test_params, P_list, B, n_test,
                      rng_test, device=device)

np.savez('results/' + file_name, **results)
