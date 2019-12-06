#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy

class AbstractCost(object):
    
    def __init__(self):
        self.gamma = 1.                     # it is changed by gamma * AbstractCost
    
    def value_expr(self, h_plus, u): # add t for dynamic adjust
        raise NotImplementedError
    
    def __mul__(self, other):
        new_obj = copy.copy(self)
        new_obj.gamma *= other
        return new_obj