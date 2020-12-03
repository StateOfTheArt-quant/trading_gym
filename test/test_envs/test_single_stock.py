import pandas as pd
import numpy as np
from trading_gym.envs.portfolio_gym.portfolio_gym import PortfolioTradingGym

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


def test_single_number():
    order_book_ids = ["000001.XSHE"]
    mock_data = create_mock_data(order_book_ids=order_book_ids, start_date="2019-01-01", end_date="2019-01-14")
    sequence_window = 1
    env = PortfolioTradingGym(data_df=mock_data, sequence_window=sequence_window, add_cash=True)
    state = env.reset()
    h_t_list = []
    orderlist = [0.5, 0.8, 0.4, 1.0, 0.0]
    for i in range(len(orderlist)):
        next_state, reward, done, info = env.step(orderlist[i])
        h_t_list.append(info["h_t"])
    '''
    stock_reward = [0.0219, -0.0103, 0.0175, -0.0017, -0.0039, 0.0059]
    v_t = portfolio_value = [1000000, 994900.0, 1008848.5, 1008223.0, 1004290.9, 1004391.4]
    expected_portfolio calculate: portfolio_value[i]*weight*mock_data["returns"][i+1] for stock and cash
     000001.XSHE    2019-01-01    0.0219
               2019-01-02   -0.0103
               2019-01-03    0.0175
               2019-01-04   -0.0017
               2019-01-05   -0.0039
               2019-01-06    0.0059
               2019-01-07   -0.0049
               2019-01-08   -0.0003
               2019-01-09   -0.0136
               2019-01-10    0.0068
               2019-01-11    0.0077
               2019-01-12    0.0136
               2019-01-13   -0.0022
               2019-01-14   -0.0012
    '''
    expected_portfolio = ([494850, 500050], [809848.6, 198999.898], [402853.3822, 605369.6309], [1004290.9, 0], [0, 1004391.359])
    np.testing.assert_almost_equal(h_t_list, expected_portfolio, decimal=1)


def test_single_array():
    order_book_ids = ["000001.XSHE"]
    mock_data = create_mock_data(order_book_ids=order_book_ids, start_date="2019-01-01", end_date="2019-01-14")
    sequence_window = 1
    env = PortfolioTradingGym(data_df=mock_data, sequence_window=sequence_window, add_cash=True)
    state = env.reset()
    h_t_list = []
    orderlist = [0.5, 0.8, 0.4, 1.0, 0.0]
    for i in range(len(orderlist)):
        next_state, reward, done, info = env.step((orderlist[i], 0))
        h_t_list.append(info["h_t"])
    '''
    Action for "cash" is never used. Whenever add_cash = true, next "cash" is calculated by negative sum of the other 
    trade (in market_simulator.py). Then action for cash can be any number and the results are still right.
    '''
    expected_portfolio = ([494850, 500050], [809848.6, 198999.898], [402853.3822, 605369.6309], [1004290.9, 0], [0, 1004391.359])
    np.testing.assert_almost_equal(h_t_list, expected_portfolio, decimal=1)


def test_single_cash_false():
    order_book_ids = ["000001.XSHE"]
    mock_data = create_mock_data(order_book_ids=order_book_ids, start_date="2019-01-01", end_date="2019-01-14")
    sequence_window = 1
    env = PortfolioTradingGym(data_df=mock_data, sequence_window=sequence_window, add_cash=False)
    state = env.reset()
    h_t_list = []
    orderlist = [0.5, 0.8, 0.4, 1.0, 0.0]
    for i in range(len(orderlist)):
        next_state, reward, done, info = env.step(orderlist[i])
        h_t_list.append(info["h_t"])
    '''
    Portfolio value becomes stock value incorrectly after every step.
    When portfolio value = 0, self.w_t (h_next / sum(h_next) in portfolio_gym) is NaN, then trigger value error in 
    market simulator. 
    '''
    expected_portfolio = ([494850], [809848.6], [402853.3822], [1004290.9], [0])
    np.testing.assert_almost_equal(h_t_list, expected_portfolio, decimal=1)


if __name__ == "__main__":
    # test_single_number()
    test_single_array()
    # test_single_cash_false()
    '''
    In single stock case, single number, array or list all work. List or array need a room for "cash" in series
    (portfolio_gym, w_t_plus), but it could be any number in case it is never used. Single number will give same
    weight for the series, it still works because of same reason (weight for cash is never used)
    Note that the three def above can only run a single one at same time. It is for the correctness of assertion.
    '''