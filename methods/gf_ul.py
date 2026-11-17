# -*- coding: utf-8 -*-
'''
Proposed GF UL calibration (joint TX and RX optimization) with omega_r ~ U[0, 1] and:
    - eta_r = omega_r (default), or a fixed eta_r with --etaFixed (Figs. 4 and 6: 0.0 for only
      communication, 1.0 for only sensing).
    - Optionally, known TX impairments with --knownTx 1 (Fig. 4): Psi_tx is set to the true
      Xi_tx and is not trained, so that only the RX parameters Psi_rx are calibrated.
    - Optionally, clutter in the training sensing channel with --clutterVar sigma_c2 (Fig. 5):
      C ~ U{0, ..., C_max} clutter scatterers with gains ~ CN(sigma_c2). Change sigma_c2 to sweep
      the SCNR of eq. (5) (saved as scnr_s). Testing is clutter-free.
'''
from ..lib.simulation_parameters import *

clutter = args.clutterVar is not None
C_max = 5 if clutter else None
sigma_c2 = torch.tensor(args.clutterVar, device=device) if clutter else 0.0

known_tx = bool(args.knownTx)
train_tx_mask = None
if known_tx:
    network.xi_tx.data = torch.clone(gamma_tx)
    network.omega_tx.data = torch.clone(p_tx)
    train_tx_mask = torch.zeros(train_it, dtype=torch.bool, device=device)

omega_r = torch.rand(train_it, device=device)
eta_r = torch.clone(omega_r) if args.etaFixed is None else args.etaFixed*torch.ones(train_it, device=device)

method = 'gf_ul_clutter_' + '{:.2e}'.format(sigma_c2.item()) if clutter else 'gf_ul'
method += '_known_tx' if known_tx else ''
file_name = fileName(method, eta_r, P, train_it, impaired, ref_sensor, seed,
                     sigma=sigma, sens_loss=args.sensLoss, I_omp=args.ompIt)
print(file_name, flush=True)

### Training ###
loss_r, loss_c, scnr_s = train(network, 'gf_ul', system, sens, comm, B, train_it, omega_r, eta_r,
                               opt_tx, sch_tx, optimizer_rx=opt_rx, scheduler_rx=sch_rx, sigma=sigma,
                               sens_loss=args.sensLoss, I_omp=args.ompIt, train_tx_mask=train_tx_mask,
                               C_max=C_max, sigma_c2=sigma_c2, rng_gen=rng_train, device=device)
saveNetwork('models/' + file_name + '_model', network)

### Testing ###
testing = replace(test_params, delta_init=torch.linspace(1.15e-2, 1.2e-2, 3, device=device))
results = evaluateNetwork(network, system, sens, comm, comm_isac, testing, test_params, R_max_c_sweep, R_min_c,
                          B, n_test, n_test_comm, rng_test, device=device)

clutter_params = {'C_max': C_max, 'sigma_c2': sigma_c2.cpu().detach().numpy()} if clutter else {}
np.savez('results/' + file_name,
         omega_r=omega_r.cpu().detach().numpy(), eta_r=eta_r.cpu().detach().numpy(),
         **clutter_params, loss_r=loss_r, loss_c=loss_c, scnr_s=scnr_s,
         **results, **impairmentsToSave(network, system))
