# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from trading_gym.envs.portfolio_gym.portfolio_gym import PortfolioTradingGym

# ============================================= #
# todo:                                         #
# ============================================= #

np.random.seed(647)


def create_mock_data(order_book_ids, start_date="2019-01-01", end_date="2019-06-11", number_feature=3):
    trading_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    number = len(trading_dates) * len(order_book_ids)
    multi_index = pd.MultiIndex.from_product([order_book_ids, trading_dates], names=["order_book_id", "datetime"])
    mock_data = pd.DataFrame(np.random.randn(number, number_feature + 1), index=multi_index,
                             columns=["feature1", "feature2", "feature3", "returns"])
    mock_data["returns"] = mock_data["returns"] / 100
    return mock_data


def test_one_stock():

    order_book_ids = ["000001.XSHE"]
    mock_data = create_mock_data(order_book_ids=order_book_ids, start_date="2019-04-01", end_date="2019-6-5")
    sequence_window = 1
    env = PortfolioTradingGym(data_df=mock_data, sequence_window=sequence_window, add_cash=False)

    observation = env.reset()
    print(observation, "observation")
    action = np.array([1.])


    while True:
        next_state, reward, done, info = env.step(action)
        if done:
            break

    portfolio_reward = np.array(env.experience_buffer["reward"])
    expected_reward = mock_data.xs("000001.XSHE")["returns"].iloc[sequence_window:].values
    np.testing.assert_almost_equal(portfolio_reward, expected_reward, decimal=3)    # 比较是否有偏差
    env.render()

    fee = 0.999  # 长期持有中，扣除手续费后与市场的对比
    portfolio_returns = portfolio_reward*fee - 1
    market_return = expected_reward - 1
    difference = portfolio_returns - market_return
    results = np.vstack((portfolio_returns, market_return, difference))
    print(results)
    print(difference)


if __name__ == "__main__":
    test_one_stock()


