#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from featurizer.interface import Functor
from featurizer.functors.labeler import RegressionLabler
import featurizer.functors.fundamental as fd
import featurizer.functors.volume_price as vp
import featurizer.functors.journalhub as jf
import featurizer.functors.time_series as tf

# ============================================================== #
# step1: create your feature definition                          #
# ============================================================== #
class CustomFeature(Functor):
    def __init__(self):
        pass
    
    def forward(self, net_profit_ts, capitalization_ts):
        return net_profit_ts/capitalization_ts

class AllFeature(object):
    
    def __init__(self):
                
        self.pct_change = tf.PctChange(window=1)
        self.ReturnsRollingStd = jf.ReturnsRollingStd(window=4)
        self.net_profit = CustomFeature()
    
    def forward(self,close_ts, net_profit_ts, capitalization_ts):
        
        feature_list = []
        feature_name_list = []
        
        returns_ts = self.pct_change(close_ts)
        feature1 = self.ReturnsRollingStd(returns_ts)
        feature2 = self.net_profit(net_profit_ts, capitalization_ts)
        
        feature_list.extend([feature1, feature2, returns_ts])
        feature_name_list.extend(["ReturnsRollingStd", "net_profit", "returns"])
        return feature_list, feature_name_list


        
# ======================================================================= #
# step2: get data                                                         #
# ======================================================================= #
import os
import jqdatasdk
from xqdata.api import history_bars

jqdata_username = os.environ["JQDATA_USERNAME"]
jqdata_password = os.environ["JQDATA_PASSWORD"]
jqdatasdk.auth(username=jqdata_username, password=jqdata_password)


order_book_ids = ['000001.XSHE','600000.XSHG']
all_fields = ["close", "net_profit", "capitalization"]
bar_count=40
dt = "2019-08-20"
frequency="1d"


data_df = history_bars(order_book_ids=order_book_ids, bar_count=bar_count, frequency=frequency, fields=all_fields, dt=dt)


# =================================================================== #
# step3: create feature                                               #
# =================================================================== #
import torch
import pandas as pd

def create_raw_feature(raw_data: pd.DataFrame) -> pd.DataFrame :  

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    close_ts = torch.tensor(raw_data["close"].unstack(0).values, dtype=torch.float32, device=device)
    net_profit_ts = torch.tensor(raw_data["net_profit"].unstack(0).values, dtype=torch.float32, device=device)
    capitalization_ts = torch.tensor(raw_data["capitalization"].unstack(0).values, dtype=torch.float32, device=device)
    
    featurizer = AllFeature()
    feature_list, feature_name_list = featurizer.forward(close_ts, net_profit_ts, capitalization_ts)
    
    data_container = {}
    for i, feature in enumerate(feature_list):
        raw_feature_df = pd.DataFrame(feature.cpu().numpy(), index=raw_data.index.levels[1], columns=raw_data.index.levels[0])
        data_container[feature_name_list[i]] = raw_feature_df
        
    featured_df = pd.concat(data_container)
    featured_df = featured_df.stack(0).unstack(0).swaplevel(0,1).sort_index(level=0)
    return featured_df.rename_axis(index=["order_book_id", "datetime"])


feature_df = create_raw_feature(data_df)


# ================================================================ #
# step4: create portfolio trading gym                              #
# ================================================================ #
from trading_gym.envs.portfolio_gym.data_generator import DataGeneratorDF

data_generator = DataGeneratorDF(data_df=feature_df,sequence_window=2, add_cash=True)

observation = data_generator.reset()

next_observation, one_step_fwd_returns, dt = data_generator.step()



