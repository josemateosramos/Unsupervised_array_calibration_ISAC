# -*- coding: utf-8 -*-
'''
Model-based communication baseline (Sec. IV-C1): SNR and SER for increasing maximum UE range,
with the TX impairments assumed by the BS.
'''
from ..lib.simulation_parameters import *

file_name = 'baseline_comm' + commonTag(P, impaired, ref_sensor, seed)
print(file_name, flush=True)

A_tx = txDictionary(xi_tx_init, omega_tx_init, theta_grid, lamb)
snr_c, ser = runCommSweep(system, comm, A_tx, R_max_c_sweep, R_min_c, B, n_test_comm, rng_test, device=device)
print('SNR: ', snr_c)
print('SER: ', ser)

np.savez('results/' + file_name, snr_c_sweep=snr_c, ser_sweep=ser)
