#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from trading_gym.utils.data.toy import create_toy_data
from trading_gym.envs.portfolio_gym.portfolio_gym import PortfolioTradingGym

order_book_id_number = 100
toy_data = create_toy_data(order_book_ids_number=order_book_id_number, feature_number=10, start="2019-05-01", end="2019-12-12", frequency="D")

env = PortfolioTradingGym(data_df=toy_data, sequence_window=1, add_cash=False)
state = env.reset()

while True:
    next_state, reward, done, info = env.step(action=None)
    label = info["one_step_fwd_returns"]
    print(state)
    print(label)
    
    #
    regressor = LinearRegression()
    regressor.fit(state.values, label.values)
    
    #display and store
    print(regressor.coef_)
    env.experience_buffer["coef"].append(regressor.coef_)
    #
    if done:
        break
    else:
        state = next_state
#
factor_returns = pd.DataFrame(np.array(env.experience_buffer["coef"]), index=env.experience_buffer["dt"], columns=toy_data.columns[:-1])

cum_factor_returns = (factor_returns +1).cumprod()
cum_factor_returns.plot(title="Cumulative Factor Return",linewidth=2.2)

