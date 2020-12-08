#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from trading_gym.interface import AbstractCost
from trading_gym.envs.portfolio_gym.costs import TCostModel
import pdb


class MarketSimulator(object):
    
    def __init__(self, costs=[], cash_key="CASH"):
        if costs is None:
            costs = []
        self.cash_key = cash_key
        # for cost in costs:
        #     assert isinstance(cost, AbstractCost)
        self.costs = costs
        
    def step(self, h, u, one_step_fwd_returns):
        """forward step the portfolio forward over the time period t, given trades u
        
        Args:
            h: [pd.Series], describe current portfolio
            u: [pd.Series], n vector with the stock trades (not cash)
            t: current time
        Returns:
            h_next: portfolio after returns propagation
            u: trades vector with simulated cash balance
        """
        h_plus = h + u
        costs = [cost.value_expr(h_plus=h_plus, u=u) for cost in self.costs]

        for cost in costs[0]: # cost is a pd.Series
            assert (not pd.isnull(cost))
            assert (not np.isinf(cost))
        
        if self.cash_key in h.index:
            u[self.cash_key] = - sum(u[u.index != self.cash_key]) - sum(sum(costs))
            h_plus[self.cash_key] = h[self.cash_key] + u[self.cash_key]
        else:
            for cost in costs:   # this is a more general form, the key is the output of cost.value_expr
                h_plus -= cost
        h_next = one_step_fwd_returns * h_plus + h_plus

        try:
            assert (not h_next.isnull().values.any())
            assert (not u.isnull().values.any())
        except Exception as e:
            print("h:{}".format(h))
            print("u:{}".format(u))
            print("h_plus:{}".format(h_plus))
            print("one_step_fwd_returns: {}".format(one_step_fwd_returns))
            print("h_next: {}".format(h_next))
            print("u: {}".format(u))
            raise ValueError("error")
        return h_next
    
    def reset(self):
        pass
    
    