import pandas as pd
import numpy as np
import gym
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
from trading_gym.envs.portfolio_gym.data_generator import DataGeneratorDF
from trading_gym.envs.portfolio_gym.market_simulator import MarketSimulator

sns.set(rc={'figure.figsize':(10,8.27)})


class PortfolioTradingGym(gym.Env):
    """
    An environment for financial portfolio management.
    """
    metadata = {'render.modes': ['human', 'ansi']}
    
    
    def __init__(self, data_df, sequence_window=5, portfolio_value=1000000, add_cash=False):
        
        self.order_book_ids = list(data_df.index.levels[0])
        self.number_order_book_ids = len(self.order_book_ids)
        self.add_cash = add_cash
        if add_cash:
            self.order_book_ids = self.order_book_ids + ["CASH"]
        self.portfolio_value = portfolio_value
        
        self.data_generator = DataGeneratorDF(data_df=data_df, sequence_window=sequence_window, add_cash=add_cash)
        self.market_simulator = MarketSimulator()   
        
        #
        self.action_space = gym.spaces.Box(0, 1, shape=(len(self.order_book_ids),), dtype=np.float32)  # include cash

        # get the state space from the data min and max
        if sequence_window == 1:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.order_book_ids), self.data_generator.number_feature), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.order_book_ids), sequence_window, self.data_generator.number_feature), dtype=np.float32)
    
    def step(self, action):
        
        next_state, one_step_fwd_returns, dt, done= self.data_generator.step()
        info ={}
        info["dt"] =dt
        info["one_step_fwd_returns"] = one_step_fwd_returns
        self.experience_buffer["dt"].append(dt)
        
        if action is None:
            return next_state, None, done, info
        
        w_t_plus = pd.Series(action, index=self.order_book_ids)
        z_t = w_t_plus - self.w_t
        u_t = z_t * self.v_t

        try:
            h_next = self.market_simulator.step(h=self.h_t, u=u_t, one_step_fwd_returns=one_step_fwd_returns)
        except Exception:
            print("dt:{}".format(dt))
            print("action:{}".format(action))
            print("self.w_t:{}".format(self.w_t))
            print("w_t_plus :{}".format(w_t_plus))
            print("v_t {}, u_t:{}".format(self.v_t, u_t))
            raise ValueError("eeee")

        info["h_t"] = h_next
        v_t_1 = sum(h_next)
        reward = (v_t_1 - self.v_t)/self.v_t
        # update
        self.w_t = h_next / sum(h_next) # that's how NaN comes from
        self.v_t = v_t_1
        self.h_t = h_next
        
        self.experience_buffer["reward"].append(reward)
        if self.add_cash:
            reward_benchmark = one_step_fwd_returns.iloc[:-1].mean()
        else:
            reward_benchmark = one_step_fwd_returns.mean()
        self.experience_buffer["reward_benchmark"].append(reward_benchmark)        
        return next_state, reward, done, info
        
    def reset(self):
        return self._reset()
    
    def _reset(self):
        
        self.experience_buffer = defaultdict(list) 
        
        if self.add_cash:
            self.w_t = pd.Series([0]*self.number_order_book_ids+[1.], index=self.order_book_ids)
        else:
            self.w_t = pd.Series([1.]*self.number_order_book_ids, index=self.order_book_ids)/self.number_order_book_ids
        self.v_t = self.portfolio_value
        self.h_t = self.w_t * self.v_t
        
        state = self.data_generator.reset()
        self.market_simulator.reset()

        return state
    
    def render(self, mode="human"):
        df = pd.DataFrame({"reward":self.experience_buffer["reward"], "reward_benchmark": self.experience_buffer["reward_benchmark"]}, index=self.experience_buffer["dt"])
        
        df["strategy_portfolio_net_value"] = (df["reward"] + 1).cumprod()
        df["equal_weighted_benchmark"] = (df["reward_benchmark"] + 1).cumprod()
        
        sns.lineplot(data=df[["strategy_portfolio_net_value","equal_weighted_benchmark"]], palette=sns.color_palette(["#01386a","#960056"]),linewidth=2.4, dashes=False)
        plt.xlabel("datetime")
        plt.ylabel("unit net value ($)")
        plt.show()

