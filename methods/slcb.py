# -*- coding: utf-8 -*-
'''
SLCB baseline (Sec. IV-C2): supervised losses (40)-(41) with the true channel gradient and
eta_r = omega_r ~ U[0, 1]. Training draws from the global RNG.
'''
from ..lib.simulation_parameters import *

omega_r = torch.rand(train_it, device=device)
eta_r = torch.clone(omega_r)

file_name = fileName('slcb', eta_r, P, train_it, impaired, ref_sensor, seed)
print(file_name, flush=True)

### Training ###
loss_r, loss_c, scnr_s = train(network, 'slcb', system, sens, comm, B, train_it, omega_r, eta_r,
                               opt_slcb, sch_slcb, rng_gen=None, device=device)
saveNetwork('models/' + file_name + '_model', network)

### Testing ###
testing = replace(test_params, delta_init=torch.linspace(1.15e-2, 1.2e-2, 3, device=device))
results = evaluateNetwork(network, system, sens, comm, comm_isac, testing, test_params, R_max_c_sweep, R_min_c,
                          B, n_test, n_test_comm, rng_test, device=device)

np.savez('results/' + file_name,
         omega_r=omega_r.cpu().detach().numpy(), eta_r=eta_r.cpu().detach().numpy(),
         loss_r=loss_r, loss_c=loss_c, scnr_s=scnr_s,
         **results, **impairmentsToSave(network, system))
