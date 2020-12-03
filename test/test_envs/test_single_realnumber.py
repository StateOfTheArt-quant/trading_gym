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


def test_single_array_R():
    order_book_ids = ["000001.XSHE"]
    mock_data = create_mock_data(order_book_ids=order_book_ids, start_date="2019-01-01", end_date="2019-01-14")
    sequence_window = 1
    env = PortfolioTradingGym(data_df=mock_data, sequence_window=sequence_window, add_cash=True)
    state = env.reset()
    h_t_list = []
    orderlist = [0.5, 0.8, 0.4, 1.0, 0.0, -0.5, -0.7, -0.4, -1.0, 0.3]
    for i in range(len(orderlist)):
        next_state, reward, done, info = env.step([orderlist[i], 0])
        h_t_list.append(info["h_t"])
    '''
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
    expected_h_t = ([494850, 500050], [809848.6, 198999.898], [402853.3822, 605369.6309],
                    [1004290.9, 0], [0, 1004391.359], [-499734.9212, 1506737.699], [-704690.4898, 1712075.95],
                    [-397473.9834, 1410480.594], [-1019895.146, 2026216.001], [304220.9, 704495.1425])
    np.testing.assert_almost_equal(h_t_list, expected_h_t, decimal=1)


def test_single_number_R():
    order_book_ids = ["000001.XSHE"]
    mock_data = create_mock_data(order_book_ids=order_book_ids, start_date="2019-01-01", end_date="2019-01-14")
    sequence_window = 1
    env = PortfolioTradingGym(data_df=mock_data, sequence_window=sequence_window, add_cash=True)
    state = env.reset()
    h_t_list = []
    orderlist = [0.5, 0.8, 0.4, 1.0, 0.0, -0.5, -0.7, -0.4, -1.0, 0.3]
    for i in range(len(orderlist)):
        next_state, reward, done, info = env.step(orderlist[i])
        h_t_list.append(info["h_t"])
    expected_h_t = ([494850, 500050], [809848.6, 198999.898], [402853.3822, 605369.6309],
                    [1004290.9, 0], [0, 1004391.359], [-499734.9212, 1506737.699], [-704690.4898, 1712075.95],
                    [-397473.9834, 1410480.594], [-1019895.146, 2026216.001], [304220.9, 704495.1425])
    np.testing.assert_almost_equal(h_t_list, expected_h_t, decimal=1)


def test_single_cashfalse_R():
    order_book_ids = ["000001.XSHE"]
    mock_data = create_mock_data(order_book_ids=order_book_ids, start_date="2019-01-01", end_date="2019-01-14")
    sequence_window = 1
    env = PortfolioTradingGym(data_df=mock_data, sequence_window=sequence_window, add_cash=False)
    state = env.reset()
    h_t_list = []
    orderlist = [0.5, 0.8, 0.4, 1.0, 0.0, -0.5, -0.7, -0.4, -1.0, 0.3]
    for i in range(len(orderlist)):
        next_state, reward, done, info = env.step(orderlist[i])
        h_t_list.append(info["h_t"])
    expected_h_t = ([494850], [809848.6], [402853.3822],[1004290.9], [0], [-499734.9212], [-704690.4898],
                    [-397473.9834], [-1019895.146], [304220.9])
    np.testing.assert_almost_equal(h_t_list, expected_h_t, decimal=1)


if __name__ == "__main__":
    #test_single_array_R()
    test_single_number_R()
    #test_single_cashfalse_R()
    '''
    Results here are same as those in test_single_stock.py.
    '''