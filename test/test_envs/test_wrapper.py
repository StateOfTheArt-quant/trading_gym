#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import torch
from trading_gym.utils.data.toy import create_toy_data
from trading_gym.envs import PortfolioTradingGym
from trading_gym.wrapper import Numpy, Torch

class TestEnvWrapper(unittest.TestCase):
    def setUp(self):
        self.order_book_id_number = 100
        self.feature_number = 10
        self.sequence_window = 3
        toy_data = create_toy_data(order_book_ids_number=self.order_book_id_number, feature_number=self.feature_number, start="2019-05-01", end="2019-12-12", frequency="D")
        env = PortfolioTradingGym(data_df=toy_data, sequence_window=self.sequence_window, add_cash=False)
        self.env = env
    
    def test_numpy_wrapper(self):
        env = Numpy(self.env)
        state = env.reset()
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (self.order_book_id_number, self.sequence_window, self.feature_number))
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
    
    def test_torch_wrapper(self):
        env = Torch(self.env)
        state_ts = env.reset()
        self.assertIsInstance(state_ts, torch.Tensor)
        self.assertEqual(state_ts.shape, (self.order_book_id_number, self.sequence_window, self.feature_number))
        
        action = env.action_space.sample()
        action = action/action.sum()
        action_ts = torch.tensor(action)
        while True:
            next_state, reward, done, info = env.step(action=action_ts)
            #
            if done:
                break
            else:
                state = next_state
        env.render()

if __name__ == "__main__":
    unittest.main()