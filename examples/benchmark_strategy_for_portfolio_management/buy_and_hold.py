import numpy as np
import cherry as ch
from trading_gym.utils.data.toy import create_toy_data
from trading_gym.envs import PortfolioTradingGym
from trading_gym.wrapper import Numpy

np.random.seed(520)

order_book_id_number = 3
toy_data = create_toy_data(order_book_ids_number=order_book_id_number, feature_number=20, start="2019-05-01", end="2019-12-12", frequency="D")
env = PortfolioTradingGym(data_df=toy_data, sequence_window=5, add_cash=False)
env = Numpy(env)

state = env.reset()
 
while True:
    # the init weight is equally weighted
    # after that, buy_and_hold means the next portfolio weight is equal the current weight after price chaging
    action = env.w_t.values 
    print(action)      
    next_state, reward, done, info = env.step(action)
    if done:
        break    
env.render()