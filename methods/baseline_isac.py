# -*- coding: utf-8 -*-
'''
Model-based ISAC baseline (Sec. IV-C1): ISAC trade-off over omega_r with the impairments
assumed by the BS.
'''
from ..lib.simulation_parameters import *

file_name = 'baseline_isac' + commonTag(P, impaired, ref_sensor, seed)
print(file_name, flush=True)

A_tx = txDictionary(xi_tx_init, omega_tx_init, theta_grid, lamb)
if impaired:
    testing = replace(test_params, delta_init=torch.linspace(7.8e-4, 1.1e-3, 3, device=device))
else:
    testing = replace(test_params, delta_init=torch.linspace(4.8e-4, 6.3e-4, 3, device=device))
p_d_isac, p_fa_isac, gospa_isac, ser_isac, snr_c_isac = runISAC(system, sens, comm_isac, A_tx, xi_rx_init,
                                                                omega_rx_init, testing, B, n_test, device=device)

np.savez('results/' + file_name,
         p_d_isac=p_d_isac, p_fa_isac=p_fa_isac, gospa_isac=gospa_isac,
         ser_isac=ser_isac, snr_c_isac=snr_c_isac, n_test=n_test)
