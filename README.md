# trading_gym
**trading_gym** is a unified environment for supervised learning and reinforcement learning in the context of quantitative trading.

# Philosophy

**trading_gym** is designed with the idea that, in the context of quantitative trading, different data format is needed for different research task. For example, cross-sectional data is used for explaining the cross-sectional variation in stock returns, time series data is used for timing strategy development,
sequential data is used for sequencial-model, e.g. RNN and it variation algorithm. Besides, supervised learning algorithm and reinforcement learning need different data architecture.

The goal of trading_gym is to provide a unified environment for supervised learning and reinforcement learning on top of reinforcement learning concepts framework.
> The main concepts of RL are the agent and the environment. The environment is the world that the agent lives in and interacts with. At every step of interaction, the agent sees a (possibly partial) observation of the state of the world, and then decides on an action to take. then the agent perceives a signal from the environment, some infomation that tells it how good or bad the current world state is.

As such, the trading_gym has been written with a few philosophical goals in mind:

* Easier use for supervised learning and reinforcement learning
* Easier use for sequencial and non-sequencial data architecture
* low-level data architecture to faciliate customised preprocessing

trading_gum is attempt to remove the boundary between supervised learing and reinforcement learning, and want researcher who don't do finacial research for a living to be able to use our APIs.

# Example
~~~
order_book_ids = ["000001.XSHE","600000.XSHG"]
mock_data = create_mock_data(order_book_ids=order_book_ids, start_date="2019-01-01", end_date="2019-06-11")
env = PortfolioTradingGym(data_df=mock_data, sequence_window=3, add_cash=True)
    
state = env.reset()
print(state)
action = np.array([0.6, 0.4, 0])
while True:        
    next_state, reward, done, info = env.step(action)
    if done:
        break    
env.render()
~~~

~~~
                          feature1  feature2  feature3
order_book_id datetime                                
000001.XSHE   2019-01-01  2.689461  1.648796  0.464287
              2019-01-02 -0.647372  0.306223 -1.137051
              2019-01-03  0.394545 -1.374702 -0.121645
600000.XSHG   2019-01-01  0.403601  1.313482 -2.010203
              2019-01-02 -0.261228 -1.039094  0.809173
              2019-01-03 -0.659177  0.904236  1.019546
CASH          2019-01-01  1.000000  1.000000  1.000000
              2019-01-02  1.000000  1.000000  1.000000
              2019-01-03  1.000000  1.000000  1.000000
~~~
![](https://github.com/StateOfTheArt-quant/trading_gym/blob/master/assets/images/benchmark.png)

# Install
~~~
git clone https://github.com/StateOfTheArt-quant/trading_gym.git
cd trading_gym
python setup.py install
~~~


# Examples
using trading_gym, we provide several examples to to display how the supervised learning and reinforcement learning can be unified under a framework.

### linear regression
linear regression is an essential algorithm in the context of quantitative trading which can be used for calculating factor returns, portfolio covariance estimation and estimated returns.
* [how to do linear regression in trading-gym and then calculate cumulative factor returns](examples/linear_regression/01_linear_regression.md)

### RNN and its variant

### deep reinforcement learning

# Author
Allen Yu (yujiangallen@126.com)

# License
This project following Apache 2.0 License as written in LICENSE file

Copyright 2018 Allen Yu, StateOfTheArt.quant