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
    print(mock_data["returns"])
    return mock_data


def test_single():
    order_book_ids = ["000001.XSHE"]
    mock_data = create_mock_data(order_book_ids=order_book_ids, start_date="2019-01-01", end_date="2019-01-14")
    sequence_window = 1
    env = PortfolioTradingGym(data_df=mock_data, sequence_window=sequence_window, add_cash=True)
    state = env.reset()
    h_t_list = []
    orderlist = [0.5, 0.8, 0.4, 1.0, 0.0, -0.5, -0.7, -0.4, -1.0, 0.3]
    for i in range(10):
        next_state, reward, done, info, h_t = env.step(orderlist[i])
        h_t_list.append(h_t)
    '''
    action for "cash" is never used. Whenever add_cash = true, next "cash" is calculated by negative sum of the other 
    trade (in market_simulator.py).
    stock_reward = [0.0219, -0.0103, 0.0175, -0.0017, -0.0039, 0.0059, -0.0049, -0.0003, -0.0136, 0.0068, 0.0077]
    expected_h_t = {["cash", "000001.XSHE"], [500500, 494850], [809848.6, 198999.898], [402853.3822, 605369.6309], 
                   [0, 1004290.9], [1004391.359, 0], [1506737.699, -499734.9212], [1712075.95, -704690.4898], 
                   [14104805.94, -397473.9834], [2026216.001, -1019895.146], [304220.9, 704495.1425]}
    v_t = portfolio_value = [1000000, 994900.0, 1008848.5, 1008223.0, 1004290.9, 1004391.4, 1007002.8, 1007385.4, 
                            1013006.7, 1006320.8, 1008715.9]
    '''
    print(h_t_list)


def test_mul_array(orderb,i=0.5):
    order_book_ids = ["000001.XSHE", "000002.XSHE"]
    mock_data = create_mock_data(order_book_ids=order_book_ids, start_date="2019-01-01", end_date="2019-01-13")
    sequence_window = 1
    env = PortfolioTradingGym(data_df=mock_data, sequence_window=sequence_window, add_cash=True)
    state = env.reset()
    h_t_list = []
    orderlist = [0.5, 0.8, 0.4, 1.0, 0.0, -0.5, -0.7, -0.4, -1.0, 0.3]
    next_state, reward, done, info, h_t = env.step([0.0, 0.0, 0.0])
    h_t_list.append(h_t)
    next_state, reward, done, info, h_t = env.step([0.0, i, 0])
    h_t_list.append(h_t)
    for i in range(10):
        next_state, reward, done, info, h_t = env.step([orderlist[i], orderb, 0])
        h_t_list.append(h_t)
    print(h_t_list)
    '''
    stock_action = [0.5, -0.5, 1.0, -1.0, 0.2, -0.2, -0.5, 0.5]
    stock_tradeday_reward = [-0.0030, -0.0011, 0.0271, 0.0055, -0.0178, -0.0258, 0.0203, -0.0002]
    expected_tradeday_reward = [501017.67, -500124.25, 1028692, -1013645, 196184.6, 195258.9, -513593.84, 499850]
    '''

if __name__ == "__main__":
    orderlistb = [0.5, 1.0, 0.2, -0.5]
    for i in range(4):
        test_mul_array(orderlistb[i], i=0.5)
        test_mul_array(-orderlistb[i], i=-0.5)
    test_single()

