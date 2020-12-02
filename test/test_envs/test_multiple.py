import unittest
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
    print(mock_data["returns"])
    return mock_data


def test_positive():
    order_book_ids = ["000001.XSHE", "000002.XSHE"]
    mock_data = create_mock_data(order_book_ids=order_book_ids, start_date="2019-01-01", end_date="2019-01-14")
    sequence_window = 1
    env = PortfolioTradingGym(data_df=mock_data, sequence_window=sequence_window, add_cash=True)
    state = env.reset()
    h_t_list = []
    orderlista = [0.5, 0.6, 0.2, 1.0, 0.0, 0.0, 0.4, 0.1]
    orderlistb = [0.5, 0.1, 0.7, 0.0, 1.0, 0.0, 0.6, 0.2]
    for i in range(len(orderlista)):
        next_state, reward, done, info = env.step([orderlista[i], orderlistb[i], 0])
        h_t_list.append(info["portfolio"])
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
    000002.XSHE    2019-01-01   -0.0186
                   2019-01-02    0.0098
                   2019-01-03   -0.0030
                   2019-01-04    0.0065
                   2019-01-05    0.0111
                   2019-01-06    0.0006
                   2019-01-07    0.0112
                   2019-01-08   -0.0304
                   2019-01-09   -0.0094
    '''
    expected_h_t = ([494850, 504900, 0.0], [610347.375, 99675.075, 299954.9925], [201652.0962, 711579.5973, 101007.84],
                    [1010283.99, 0.0, 0.0], [0.0, 1010890.17, 0.0], [0.0, 0.0, 1010991.259], [404275.1847, 588154.2748,
                    0.0], [97893.241, 196620.0124, 694770.1])
    np.testing.assert_almost_equal(h_t_list, expected_h_t, decimal=1)


if __name__ == "__main__":
    test_positive()