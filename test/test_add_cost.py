#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from trading_gym.utils.data.toy import create_toy_data
from trading_gym.envs import PortfolioTradingGym
from trading_gym.envs.portfolio_gym.costs import TCostModel
import pdb

np.random.seed(64)

commitment_fee = TCostModel(half_spread=0.01)
mock_data = create_toy_data(order_book_ids_number=2, feature_number=3, start="2019-01-01", end="2019-01-11")
'''
0001.XSHE      2019-01-01    0.0219
               2019-01-02   -0.0103
               2019-01-03    0.0175
               2019-01-04   -0.0017
               2019-01-05   -0.0039
               2019-01-06    0.0059
               2019-01-07   -0.0049
               2019-01-08   -0.0003
               2019-01-09   -0.0136
               2019-01-10    0.0068
               2019-01-11    0.0077
0002.XSHE      2019-01-01    0.0136
               2019-01-02   -0.0022
               2019-01-03   -0.0012
               2019-01-04   -0.0186
               2019-01-05    0.0098
               2019-01-06   -0.0030
               2019-01-07    0.0065
               2019-01-08    0.0111
               2019-01-09    0.0006
               2019-01-10    0.0112
               2019-01-11   -0.0304
'''


def test_add_cost():
    env = PortfolioTradingGym(data_df=mock_data, sequence_window=3, add_cash=True, costs=[commitment_fee])
    state = env.reset()
    print(state)
    action = np.array([0.6, 0.4, 0])
    h_t_list = []
    while True:
        next_state, reward, done, info = env.step(action)
        h_t_list.append(info["h_t"])
        if done:
            break
    expected_costs = ([6000, 4000], [100.566, 0.556], [31.666, 32.678])
    print(h_t_list)
    '''
    When sum of weights equals to 1, cash will be negative sum of costs today.
    '''


def test_add_cost_cash_false():
    env = PortfolioTradingGym(data_df=mock_data, sequence_window=3, add_cash=False, costs=[commitment_fee])
    state = env.reset()
    print(state)
    action = np.array([0.6, 0.4])
    h_t_list = []
    while True:
        next_state, reward, done, info = env.step(action)
        h_t_list.append(info["h_t"])
        if done:
            break
    expected_costs = ([1000, 1000], [42.4552, 42.4552], [32.448, 32.448])
    print(h_t_list)
    '''
    The first-day cost is incorrect due to initialization in portfolio_gym.reset
    '''


if __name__ == "__main__":
    #test_add_cost()
    test_add_cost_cash_false()