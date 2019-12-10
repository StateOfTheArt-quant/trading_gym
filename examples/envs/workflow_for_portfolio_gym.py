from best_practice_in_feature_engineering import feature_df

# ================================================================ #
# step4: create portfolio trading gym                              #
# ================================================================ #
from trading_gym.envs.portfolio_gym.portfolio_gym import PortfolioTradingGym

env = PortfolioTradingGym(data_df=feature_df, sequence_window=3, add_cash=True)

state = env.reset()

while True:
    action = env.action_space.sample()
    action  = action /action.sum()
    next_state, reward, done, info = env.step(action)
    print(next_state)
    if done:
        break


