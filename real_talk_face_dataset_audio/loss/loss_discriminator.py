#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 16:05 2020

@author: pati
"""

# Needed libraries and packages
import torch
import torch.nn as nn

class LossDSCreal(nn.Module):
    """
        Inputs: r
    """
    def __init__(self):
        super(LossDSCreal, self).__init__()
        self.relu = nn.ReLU()
        
    def forward(self, r):
        loss = self.relu(1.0-r)
        return loss.mean()

class LossDSCfake(nn.Module):
    """
        Inputs: rhat
    """
    def __init__(self):
        super(LossDSCfake, self).__init__()
        self.relu = nn.ReLU()
        
    def forward(self, rhat):
        loss = self.relu(1.0+rhat)
        return loss.mean()
