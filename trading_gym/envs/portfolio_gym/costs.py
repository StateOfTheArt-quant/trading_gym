#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from trading_gym.interface import AbstractCost

class TCostModel(AbstractCost):
    
    def __init__(self, half_spread, nonlin_coeff=0., sigma=0., volume=1., power=1.5, cash_key="CASH"):
        self.half_spread = half_spread
        self.nonlin_coeff = nonlin_coeff
        self.sigma = sigma
        self.volume = volume
        self.power = power
        self.cash_key = cash_key
        super(TCostModel, self).__init__()
    
    def value_expr(self, h_plus, u):
        self.tmp_tcosts = np.abs(u) * self.half_spread
        if self.cash_key in u.index:
            self.tmp_tcost[self.cash_key] = 0
        return self.tmp_tcosts
    