import pandas as pd
import numpy as np
from trading_gym.envs.portfolio_gym.portfolio_gym import PortfolioTradingGym
import pdb
np.random.seed(647)

def create_mock_data(order_book_ids, start_date="2019-01-01", end_date="2019-06-11", number_feature=3):
    
    trading_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    number = len(trading_dates) * len(order_book_ids)
    
    multi_index = pd.MultiIndex.from_product([order_book_ids, trading_dates],names=["order_book_id","datetime"])
    mock_data = pd.DataFrame(np.random.randn(number,number_feature+1), index=multi_index, columns=["feature1","feature2","feature3","returns"])
    mock_data["returns"] = mock_data["returns"]/100
    return mock_data
    
    # create multi-index Dafaframe

def test_invest_on_first_stock():
    order_book_ids = ["000001.XSHE","600000.XSHG"]
    mock_data = create_mock_data(order_book_ids=order_book_ids, start_date="2019-01-01", end_date="2019-06-11")
    
    sequence_window=3
    env = PortfolioTradingGym(data_df=mock_data, sequence_window=sequence_window, add_cash=True)
    
    state = env.reset()
    # the first stock buy and hold strategy
    action = np.array([1.0, 0.0, 0.0])
    while True:        
        next_state, reward, done, info = env.step(action)
        if done:
            break
    
    portfolio_reward = np.array(env.experience_buffer["reward"])
    expected_reward = mock_data.xs("000001.XSHE")["returns"].iloc[sequence_window:].values 
    
    #pdb.set_trace()
    np.testing.assert_almost_equal(portfolio_reward, expected_reward, decimal=3)
    
    #env.render()
    
    return mock_data, env

def test_equal_weighted_portfolio():
    order_book_ids = ["000001.XSHE","600000.XSHG"]
    mock_data = create_mock_data(order_book_ids=order_book_ids, start_date="2019-01-01", end_date="2019-06-11")
    
    sequence_window=3
    env = PortfolioTradingGym(data_df=mock_data, sequence_window=sequence_window, add_cash=True)
    
    state = env.reset()
    print(state)
    action = np.array([0.5, 0.5, 0])
    while True:        
        next_state, reward, done, info = env.step(action)
        if done:
            break
    
    portfolio_reward = np.array(env.experience_buffer["reward"])
    expected_reward = np.array(env.experience_buffer["reward_benchmark"]) 
    
    np.testing.assert_almost_equal(portfolio_reward, expected_reward, decimal=3)
    
    env.render()
    
    return mock_data, env


if __name__ == "__main__":
    test_invest_on_first_stock()
    test_equal_weighted_portfolio()
    

