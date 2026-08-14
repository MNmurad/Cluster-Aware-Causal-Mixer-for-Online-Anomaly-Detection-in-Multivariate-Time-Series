# -*- coding: utf-8 -*-
"""
Created when I should've been asleep
@author: Murad
SISLab, USF
mmurad@usf.edu
"""

import torch
import torch.nn as nn
from torch.nn import functional as F, init


class CausalLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias = True, type_ = 'causal'):
        super(CausalLinearLayer, self).__init__()
        
        self.in_features, self.out_features = in_features, out_features
        self.bias_enabled = bias
        self.weight = nn.Parameter(torch.empty(in_features, out_features)) 
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features)) 
        else:
            self.register_parameter('bias', None)
        self.register_buffer("mask", self.custom_triu(in_features, out_features, type_))
        '''
        Two type_ are available: anti_causal, causal
        causal: each neuron will depend on its previous positioned neuron
        anti_causal: each neuron will depend on the its next positioned neuron
        '''
        self.reset_parameters()

        
    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.weight)
        init.zeros_(self.bias)
    
    def forward(self, x):
        '''
        x: [*, *, in_features]
        '''
        assert len(x.shape) == 3
        upper_triangular_weight = self.weight * self.mask # [f_in, f_out]
        '''
        F.Linear(x, w, b):
            y = x * wT + b
            x-> [*, *, f_in]
            w-> [f_out, f_in]
            b-> [f_out]
            y-> [*, *, f_out]
        '''
        return F.linear(x, upper_triangular_weight.T, self.bias)

    def custom_triu(self, rows, cols, type_):
        '''
        Two type_ are available: anti_causal, causal
        causal: each neuron will depend on its previous positioned neuron
        anti_causal: each neuron will depend on the its next positioned neuron
        '''
        matrix = torch.zeros(rows, cols)
        n = rows
        for j in range(cols, 0, -1):
            matrix[:n, j-1] = 1/n # 1/n # 1/n
            n = max(n-1, 1)
        if type_ == 'causal':
            return matrix
        elif type_ == 'anti_causal':
            return create_anti_diagonal_mat(rows) @ matrix @ create_anti_diagonal_mat(cols)

def create_anti_diagonal_mat(n):
    anti_diagonal = torch.zeros(n, n)
    for i in range(n):
        anti_diagonal[i, n - 1 - i] = 1
    return anti_diagonal








