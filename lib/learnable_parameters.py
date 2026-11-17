# -*- coding: utf-8 -*-
import torch.nn as nn

class LearnableImpairments(nn.Module):
    '''Learnable parameters Psi = [xi_tx, omega_tx, xi_rx, omega_rx] of eqs. (14)-(15):
    GPIs xi_x (K,1) and antenna positions omega_x (K,).'''
    def __init__(self, xi_tx, xi_rx, omega_tx, omega_rx):
        super(LearnableImpairments, self).__init__()
        self.xi_tx = nn.Parameter(xi_tx)
        self.xi_rx = nn.Parameter(xi_rx)
        self.omega_tx = nn.Parameter(omega_tx)
        self.omega_rx = nn.Parameter(omega_rx)
