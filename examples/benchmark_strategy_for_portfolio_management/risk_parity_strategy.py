import os
import pandas as pd
import numpy as np
from trading_gym.envs.portfolio_gym.portfolio_gym import PortfolioTradingGym
import jqdatasdk
from xqdata.api import history_bars

jqdata_username = os.environ["JQDATA_USERNAME"]
jqdata_password = os.environ["JQDATA_PASSWORD"]
jqdatasdk.auth(username=jqdata_username, password=jqdata_password)
    
# ================================================== #
# step1: create environment                          #
# ================================================== #
order_book_ids = ['000001.XSHE','000002.XSHE','300520.XSHE']
fields = ["close"]
rolling_window = 20
bar_count = 200
dt = "2020-03-12"
frequency="1d"
    #
data = history_bars(order_book_ids=order_book_ids, bar_count=bar_count+rolling_window, frequency=frequency, fields=fields, dt=dt)
pct_change = data.groupby(level="order_book_id")["close"].pct_change()
pct_change.name = "returns"

feature_df = pct_change.to_frame()
feature_df = feature_df.fillna(0) 
# the first "returns" as state, the last column is feed into environment serving as to calculate next_fwd_returns
feature_df["label"] = feature_df["returns"]
env = PortfolioTradingGym(data_df=feature_df, sequence_window=20, add_cash=False)


# ============================================================================#
# step2: create a policy, policy is a funtion mapping states to action        #
# =========================================================================== #
def policy(state:pd.DataFrame) -> pd.DataFrame:
    volatility = state.groupby(level=0).std()
    volatility_reverse = 1/volatility
    weight = volatility_reverse / volatility_reverse.sum()
    weight.name = "weight"
    return weight

# =================================================== #
# step3: iterate Markov Decision Process              #
# =================================================== #
state = env.reset()
while True:
    print(state)
    action = policy(state['returns'])
    print(action)
    next_state, reward, done, info = env.step(action)
    print("next state dt:{}".format(info["dt"]))
    if done:
        break
    else:
        state = next_state
env.render()


