# -*- coding: utf-8 -*-
import numpy as np
from trading_gym.envs.portfolio_gym.portfolio_gym import PortfolioTradingGym
from trading_gym.utils.data.toy import create_toy_data


def test_one_stock():

    order_book_ids_num = 1
    order_book_ids = ["000{}.XSHE".format(order_book_ids_num)]
    mock_data = create_toy_data(order_book_ids_number=order_book_ids_num, feature_number=3, start="2019-04-01", end="2019-5-5", random_seed=1)
    print(mock_data)
    sequence_window = 1
    env = PortfolioTradingGym(data_df=mock_data, sequence_window=sequence_window, add_cash=False)

    state = env.reset()

    action = np.array([1.])


    while True:
        next_state, reward, done, info = env.step(action)
        if done:
            break

    portfolio_reward = np.array(env.experience_buffer["reward"])
    expected_reward = mock_data.xs(order_book_ids)["returns"].iloc[sequence_window:].values
    np.testing.assert_almost_equal(portfolio_reward, expected_reward, decimal=3)    # 比较是否有偏差
    env.render()


if __name__ == "__main__":
    test_one_stock()


