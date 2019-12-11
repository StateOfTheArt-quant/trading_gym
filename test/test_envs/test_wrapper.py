#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from trading_gym.utils.data.toy import create_toy_data
from trading_gym.envs import PortfolioTradingGym
from trading_gym.wrapper import Numpy, Torch

order_book_id_number = 100
toy_data = create_toy_data(order_book_ids_number=order_book_id_number, feature_number=10, start="2019-05-01", end="2019-12-12", frequency="D")

env = PortfolioTradingGym(data_df=toy_data, sequence_window=3, add_cash=False)

env = Numpy(env)

state = env.reset()
action = env.action_space.sample()
next_state, reward, done, info = env.step(action)

env = Torch(env)
state_ts = env.reset()
action = env.action_space.sample()
next_state_ts, reward, done, info_ts = env.step(action)
