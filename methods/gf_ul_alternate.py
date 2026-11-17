# -*- coding: utf-8 -*-
'''
GF UL calibration with alternate TX and RX optimization (Fig. 7): the TX is trained with the
communication loss only during the first switch_it iterations and the RX with the sensing loss
during the rest, with eta_r = omega_r ~ U[0, 1]. Set tx_first = False for the reverse order.
'''
from ..lib.simulation_parameters import *

tx_first = True

omega_r = torch.rand(train_it, device=device)
eta_r = torch.clone(omega_r)

switch_it = train_it//2
first_half = torch.arange(train_it, device=device) < switch_it
train_tx_mask = first_half if tx_first else ~first_half
train_rx_mask = ~train_tx_mask

file_name = fileName('gf_ul_alt_' + ('tx_first' if tx_first else 'rx_first'), eta_r, P, train_it,
                     impaired, ref_sensor, seed, sigma=sigma, sens_loss=args.sensLoss, I_omp=args.ompIt)
print(file_name, flush=True)

### Training ###
loss_r, loss_c, scnr_s = train(network, 'gf_ul', system, sens, comm, B, train_it, omega_r, eta_r,
                               opt_tx, sch_tx, optimizer_rx=opt_rx, scheduler_rx=sch_rx, sigma=sigma,
                               sens_loss=args.sensLoss, I_omp=args.ompIt, tx_objective='comm',
                               train_tx_mask=train_tx_mask, train_rx_mask=train_rx_mask,
                               rng_gen=rng_train, device=device)
saveNetwork('models/' + file_name + '_model', network)

### Testing ###
testing = replace(test_params, delta_init=torch.linspace(1.15e-2, 1.2e-2, 3, device=device))
results = evaluateNetwork(network, system, sens, comm, comm_isac, testing, test_params, R_max_c_sweep, R_min_c,
                          B, n_test, n_test_comm, rng_test, device=device)

np.savez('results/' + file_name,
         omega_r=omega_r.cpu().detach().numpy(), eta_r=eta_r.cpu().detach().numpy(), switch_it=switch_it,
         loss_r=loss_r, loss_c=loss_c, scnr_s=scnr_s,
         **results, **impairmentsToSave(network, system))
