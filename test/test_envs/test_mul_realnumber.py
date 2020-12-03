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


def test_mul_R():
    order_book_ids = ["000001.XSHE", "000002.XSHE"]
    mock_data = create_mock_data(order_book_ids=order_book_ids, start_date="2019-01-01", end_date="2019-01-14")
    sequence_window = 1
    env = PortfolioTradingGym(data_df=mock_data, sequence_window=sequence_window, add_cash=True)
    state = env.reset()
    h_t_list = []
    orderlista = [0.5, 0.6, -0.2, -0.3, -1.0, 1.0]
    orderlistb = [0.5, -0.2, 0.7, -0.4, 1.0, -1.0]
    for i in range(len(orderlista)):
        next_state, reward, done, info = env.step([orderlista[i], orderlistb[i], 0])
        h_t_list.append(info["h_t"])
    '''
000001.XSHE    2019-01-01    0.0219
               2019-01-02   -0.0103
               2019-01-03    0.0175
               2019-01-04   -0.0017
               2019-01-05   -0.0039
               2019-01-06    0.0059
               2019-01-07   -0.0049
000002.XSHE    2019-01-01   -0.0186
               2019-01-02    0.0098
               2019-01-03   -0.0030
               2019-01-04    0.0065
               2019-01-05    0.0111
               2019-01-06    0.0006
               2019-01-07    0.0112
    '''
    expected_h_t = ([494850, 504900, 0.0], [610347.375, -199350.15, 599909.985], [-201837.7335, 712234.6748, 505504.15],
                    [-303581.7231, -410871.0374, 1727204.558], [-1018727.033, 1013359.449, 1012853.07],
                    [1002548.843, -1018769.3595, 1007586.27])
    print(h_t_list)
    np.testing.assert_almost_equal(h_t_list, expected_h_t, decimal=1)


if __name__ == "__main__":
    test_mul_R()