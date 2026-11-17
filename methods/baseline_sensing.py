# -*- coding: utf-8 -*-
'''
Model-based sensing baseline (Sec. IV-C1): ROC and fixed-p_fa metrics with the impairments
assumed by the BS (-i 1: no impairment knowledge, -i 0: impairment knowledge).
'''
from ..lib.simulation_parameters import *

file_name = 'baseline_sensing' + commonTag(P, impaired, ref_sensor, seed)
print(file_name, flush=True)

A_tx = txDictionary(xi_tx_init, omega_tx_init, theta_grid, lamb)
p_d_roc, p_fa_roc, gospa_roc, pos_err_roc = runSensingROC(system, sens, A_tx, xi_rx_init, omega_rx_init,
                                                          getThresholdsROC(num_thr, P, device=device),
                                                          test_params, B, n_test, rng_test, device=device)
rng_test.manual_seed(1998)
p_d, p_fa, gospa_pos, pos_err = testSensingFixedPfa(system, sens, A_tx, xi_rx_init, omega_rx_init,
                                                    test_params, B, n_test, rng_gen=rng_test, device=device)

np.savez('results/' + file_name,
         p_d_roc=p_d_roc, p_fa_roc=p_fa_roc, gospa_roc=gospa_roc, pos_err_roc=pos_err_roc,
         p_d=p_d, p_fa=p_fa, gospa=gospa_pos, pos_err=pos_err)
