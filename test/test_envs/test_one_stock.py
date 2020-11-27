import pandas as pd
import numpy as np
from trading_gym.envs.portfolio_gym.portfolio_gym import PortfolioTradingGym
import random
import pdb
np.random.seed(64)

def create_mock_data(order_book_ids, start_date="2019-01-01", end_date="2022-01-02", number_feature=3):
    trading_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    number = len(trading_dates) * len(order_book_ids)
    multi_index = pd.MultiIndex.from_product([order_book_ids, trading_dates], names=["order_book_id", "datetime"])
    mock_data = pd.DataFrame(np.random.randn(number, number_feature + 1), index=multi_index,
                             columns=["feature1", "feature2", "feature3", "returns"])
    mock_data["returns"] = mock_data["returns"] / 100  # 当期收益率
    mock_data["returns"] = round(mock_data["returns"], 4)
    return mock_data

def test_base():
    order_book_ids = ["000001.XSHE"]
    mock_data = create_mock_data(order_book_ids=order_book_ids, start_date="2019-01-01", end_date="2019-01-11")
    sequence_window = 1
    env = PortfolioTradingGym(data_df=mock_data, sequence_window=sequence_window, add_cash=True)
    state = env.reset()
    h_t_list = []
    next_state, reward, done, info, h_t = env.step(0.5)
    h_t_list.append(h_t)
    h_t_list.append(500000*1.001)
    h_t_list.append(500000*(1-0.0103))
    next_state, reward, done, info, h_t = env.step(0.8)  # 正增
    h_t_list.append(h_t)
    next_state, reward, done, info, h_t = env.step(0.4)  # 正减
    h_t_list.append(h_t)
    next_state, reward, done, info, h_t = env.step(1.0)  # 边界
    h_t_list.append(h_t)
    next_state, reward, done, info, h_t = env.step(0.0)  # 边界
    h_t_list.append(h_t)
    next_state, reward, done, info, h_t = env.step(-0.5)
    h_t_list.append(h_t)
    next_state, reward, done, info, h_t = env.step(-0.7)  # 负减
    h_t_list.append(h_t)
    next_state, reward, done, info, h_t = env.step(-0.4)  # 负增
    h_t_list.append(h_t)
    next_state, reward, done, info, h_t = env.step(-1.0)  # 边界
    h_t_list.append(h_t)
    next_state, reward, done, info, h_t = env.step(0.3)  # 跨界
    h_t_list.append(h_t)
    '''
    stock_reward = [0.0219, -0.0103, 0.0175, -0.0017, -0.0039, 0.0059, -0.0049, -0.0003, -0.0136, 0.0068, 0.0077]
    expected_h_t = {["cash", "000001.XSHE"], [500500, 494850], [809848.6, 198999.898], [402853.3822, 605369.6309], 
    [0, 1004290.9], [1004391.359, 0], [1506737.699, -499734.9212], [1712075.95, -704690.4898], 
    [14104805.94, -397473.9834], [2026216.001, -1019895.146], [304220.9, 704495.1425]}
    portfolio_value [1000000, 994900.0, 1008848.5, 1008223.0, 1004290.9, 1004391.4, 1007002.8, 1007385.4, 
    1013006.7, 1006320.8, 1008715.9]
    '''
    print(h_t_list)


def test_one_stock():
    # random test
    order_book_ids = ["000001.XSHE"]
    mock_data = create_mock_data(order_book_ids=order_book_ids, start_date="2019-01-01", end_date="2020-01-02")
    sequence_window = 1
    env = PortfolioTradingGym(data_df=mock_data, sequence_window=sequence_window, add_cash=True)
    state = env.reset()
    actionlist = []

    while True:
        action = random.uniform(-1, 1)
        actionlist.append(action)
        next_state, reward, done, info, h_t = env.step(action)
        if done:
            break

    env.render()
    portfolio_reward = np.array(env.experience_buffer["reward"])
    expected_reward = mock_data.xs("000001.XSHE")["returns"].iloc[sequence_window:].values * actionlist

    for i in range(len(actionlist)):
        if actionlist[i] != 0:
            expected_reward[i] = expected_reward[i]+info["one_step_fwd_returns"][1]*(1-actionlist[i])
            np.testing.assert_almost_equal(portfolio_reward[i], expected_reward[i], decimal=6)
        else:
            np.testing.assert_almost_equal(portfolio_reward[i], info["one_step_fwd_returns"][1], decimal=4)

    return actionlist, portfolio_reward[-1]


def test_one_stock_specific():
    # specific test
    order_book_ids = ["000001.XSHE"]
    mock_data = create_mock_data(order_book_ids=order_book_ids, start_date="2019-01-01", end_date="2019-07-21")
    sequence_window = 1
    env = PortfolioTradingGym(data_df=mock_data, sequence_window=sequence_window, add_cash=True)
    state = env.reset()
    actionlist = []
    i = 0
    while True:
        action = i/100-1
        actionlist.append(action)
        next_state, reward, done, info, h_t = env.step(action)
        if action == 1:
            break
        i += 1
        if done:
            break

    env.render()
    portfolio_reward = np.array(env.experience_buffer["reward"])
    expected_reward = mock_data.xs("000001.XSHE")["returns"].iloc[sequence_window:].values * actionlist

    for i in range(len(actionlist)):
        if actionlist[i] != 0:
            expected_reward[i] = expected_reward[i] + info["one_step_fwd_returns"][1] * (1 - actionlist[i])
            np.testing.assert_almost_equal(portfolio_reward[i], expected_reward[i], decimal=6)
        else:
            np.testing.assert_almost_equal(portfolio_reward[i], info["one_step_fwd_returns"][1], decimal=4)

    return actionlist, portfolio_reward[-1]


if __name__ == "__main__":
    test_one_stock()
    test_one_stock_specific()
    test_base()

