import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from trading_gym.utils.data.toy import create_toy_data
from trading_gym.envs.portfolio_gym.portfolio_gym import PortfolioTradingGym

order_book_id_number = 100
toy_data = create_toy_data(order_book_ids_number=order_book_id_number, feature_number=10, start="2019-05-01", end="2019-12-12", frequency="D")

env1 = PortfolioTradingGym(data_df=toy_data, sequence_window=1, add_cash=False, mode="pandas")
state1 = env1.reset()

env2 = PortfolioTradingGym(data_df=toy_data, sequence_window=3, add_cash=False, mode="pandas")
state2 = env2.reset()

