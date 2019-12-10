import numpy as np
import pandas as pd
import pdb

class DataGeneratorDF(object):
    """input is a DataFrame with MultiIndex, considering Panal data structure is depreciated after pandas=0.24"""
    
    def __init__(self, data_df, sequence_window=2, add_cash=False, risk_free_return=0.0001):
        data_df = data_df.rename_axis(["order_book_id","datetime"])
        self.data_df = data_df
        self.order_book_ids = list(data_df.index.levels[0])
        self.trading_dates = list(data_df.index.levels[1])
        self.number_feature = len(data_df.columns) -1
        self.sequence_window = sequence_window
        self.add_cash = add_cash
        self.risk_free_return = risk_free_return
    
    def step(self):
        self.idx += 1
        dt = self.trading_dates[self.idx-1]
        
        if dt == self.trading_dates[-1]:    
            done = True
        else:
            done = False
            
        observation, one_step_fwd_returns = self._step(dt)
        return observation, one_step_fwd_returns, dt, done
    
    def _step(self, dt):
        idx = self.trading_dates.index(dt)+1
        trading_dates_slice = self.trading_dates[idx - self.sequence_window: idx]
        total_observation = self.data_df.loc[(self.order_book_ids, trading_dates_slice),:]
        #pdb.set_trace()
        # fillna to a balanced panel data
        total_observation = total_observation.unstack().stack(dropna=False)
        
        # observation
        observation = total_observation.iloc[:,:self.number_feature]
        # one_step_fwd_returns
        one_step_fwd_returns = total_observation.xs(dt, level="datetime").iloc[:,-1]
        one_step_fwd_returns.name = "returns at {}".format(dt)
        #pdb.set_trace()
        
        if self.add_cash:
            multi_index = pd.MultiIndex.from_tuples([("CASH", i) for i in trading_dates_slice])
            df_cash = pd.DataFrame(1, index=multi_index, columns=observation.columns)
            observation = pd.concat([observation, df_cash])
            
            one_step_fwd_returns.loc["CASH"] = self.risk_free_return
        
        
        return observation, one_step_fwd_returns
    
    
    def reset(self):
        self.idx = self.sequence_window
        first_date = self.trading_dates[self.idx-1]
        observation, one_step_fwd_returns = self._step(first_date)
        return observation


class DataGeneratorNP(object):
    def __init__(self, data_np, order_book_ids, trading_dates, sequence_window=None, risk_free_return=0.0001):
        
        number_order_book_id, total_dates, num_feature = data_np.shape
        self.number_feature = num_feature
        self._data = data_np
        self.order_book_ids = order_book_ids
        self.trading_dates = trading_dates
        self.sequence_window = sequence_window
        self.risk_free_return= risk_free_return
    
    def step(self):
        self.steps += 1
        
        self.idx = self.steps + self.sequence_window
        next_state = self._data[:,self.idx - self.sequence_window: self.idx, :self.number_feature-1]
        dt = self.trading_dates[self.idx-1]
        one_step_fwd_returns = self._data[:,self.idx-1, self.number_feature-1]
        
        
        one_step_fwd_returns = pd.Series(index=self.order_book_ids, data=one_step_fwd_returns)
        one_step_fwd_returns.loc["CASH"] = self.risk_free_return
        
        return next_state, one_step_fwd_returns, dt
    
    def reset(self):
        self.steps = 0
        self.idx = self.steps + self.sequence_window                
        dt = self.trading_dates[self.idx]
        #pdb.set_trace()
        observation = self._data[:,self.idx - self.sequence_window: self.idx, :self.number_feature-1]

        return observation 

if __name__ == "__main__":
    pass
#    order_book_ids = ["000001.XSHE","000002.XSHE"]
#    total_dates = 20
#    number_feature = 4   # contain returns
#    sequence_window = 2
#    
#    data = np.random.randn(len(order_book_ids),total_dates, number_feature)
#    
#    data_generator = DataGenerator(data_np=data, order_book_ids=order_book_ids,trading_dates=list(range(1,total_dates+1)),sequence_window=sequence_window)
#        
#    state = data_generator.reset()
#    next_state, one_step_fwd_returns, dt = data_generator.step()        
    