import pandas as pd
import numpy as np

def create_toy_data(order_book_ids_number, feature_number, start, end, frequency="D", random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    order_book_ids = ["000{}.XSHE".format(i) for i in range(1,order_book_ids_number+1)]
    
    trading_datetime = pd.date_range(start=start, end=end, freq=frequency)
    number = len(trading_datetime) * len(order_book_ids)
    
    multi_index = pd.MultiIndex.from_product([order_book_ids, trading_datetime],names=["order_book_id","datetime"])
    
    column_names = ["feature_{}".format(i) for i in range(1,feature_number+1)]
    column_names += ["returns"]
    
    toy_data = pd.DataFrame(np.random.randn(number, feature_number+1), index=multi_index, columns=column_names)
    toy_data["returns"] = round(toy_data["returns"]/100, 4)
    return toy_data
    
    