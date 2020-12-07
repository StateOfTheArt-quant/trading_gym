#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from trading_gym.utils.data.toy import create_toy_data
from trading_gym.envs import PortfolioTradingGym

from trading_gym.envs.portfolio_gym.costs import TCostModel

commitment_fee = TCostModel(half_spread=0.01)


mock_data = create_toy_data(order_book_ids_number=2, feature_number=3, start="2019-01-01", end="2019-06-11")
env = PortfolioTradingGym(data_df=mock_data, sequence_window=3, add_cash=True, costs=[commitment_fee])
    
state = env.reset()
print(state)
action = np.array([0.6, 0.4, 0])
while True:        
    next_state, reward, done, info = env.step(action)
    if done:
        break    
env.render()
