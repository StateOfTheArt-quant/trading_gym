#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from trading_gym.utils.data.toy import create_toy_data
from trading_gym.envs.portfolio_gym.portfolio_gym import PortfolioTradingGym

# ============================================= #
# todo:                                         #
# ============================================= #
order_book_id_number = 1
toy_data = create_toy_data(order_book_ids_number=order_book_id_number, feature_number=3, start="2019-05-01", end="2019-7-12", frequency="D",random_seed=123)

env = PortfolioTradingGym(data_df=toy_data, sequence_window=1, add_cash=False)

observation = env.reset()
print(observation)
action = np.array([1.])
total_steps=list(range(2))
for step in total_steps:
    next_state, reward, done, info = env.step(action)
    print(next_state, reward)


