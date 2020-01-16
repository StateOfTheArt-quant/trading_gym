import copy
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import random
import pdb

import cherry as ch
from cherry import td
from cherry import pg
from cherry import envs

from trading_gym.utils.data.toy import create_toy_data
from trading_gym.envs import PortfolioTradingGym
from trading_gym.wrapper import Numpy
from models import ActorCritic


DISCOUNT = 0.99
EPSILON = 0.05
HIDDEN_SIZE = 50
LEARNING_RATE = 0.001
MAX_STEPS = 5000
BATCH_SIZE = 1024
TRACE_DECAY = 0.97
SEED = 42
PPO_CLIP_RATIO = 0.2
PPO_EPOCHS = 20
REPLAY_SIZE = 100000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def main():
    order_book_id_number = 10
    toy_data = create_toy_data(order_book_ids_number=order_book_id_number, feature_number=20, start="2019-05-01", end="2019-12-12", frequency="D")
    env = PortfolioTradingGym(data_df=toy_data, sequence_window=5, add_cash=True)
    env = Numpy(env)
    env = ch.envs.Logger(env, interval=1000)
    env = ch.envs.Torch(env)
    env = ch.envs.Runner(env)
    
    # create net
    action_size = env.action_space.shape[0]
    number_asset, seq_window, features_number = env.observation_space.shape
    input_size = features_number
    
    agent = ActorCritic(input_size=input_size, hidden_size=HIDDEN_SIZE, action_size=action_size)
    actor_optimiser = optim.Adam(agent.actor.parameters(), lr=LEARNING_RATE)
    critic_optimiser = optim.Adam(agent.critic.parameters(), lr=LEARNING_RATE)
    
    replay = ch.ExperienceReplay()

    for step in range(1, MAX_STEPS + 1):
        replay += env.run(agent, episodes=1)

        if len(replay) >= BATCH_SIZE:
            with torch.no_grad():
                advantages = pg.generalized_advantage(DISCOUNT,
                                                      TRACE_DECAY,
                                                      replay.reward(),
                                                      replay.done(),
                                                      replay.value(),
                                                      torch.zeros(1))
                advantages = ch.normalize(advantages, epsilon=1e-8)
                returns = td.discount(DISCOUNT,
                                         replay.reward(),
                                         replay.done())
                old_log_probs = replay.log_prob()
                
            # here is to add readability    
            new_values = replay.value()
            new_log_probs = replay.log_prob()
            for epoch in range(PPO_EPOCHS):
                # Recalculate outputs for subsequent iterations
                if epoch > 0:
                    _, infos = agent(replay.state())
                    masses = infos['mass']
                    new_values = infos['value']
                    new_log_probs = masses.log_prob(replay.action()).unsqueeze(-1)
                
                # Update the policy by maximising the PPO-Clip objective
                policy_loss = ch.algorithms.ppo.policy_loss(new_log_probs,
                                                            old_log_probs,
                                                            advantages,
                                                            clip=PPO_CLIP_RATIO)
                actor_optimiser.zero_grad()
                policy_loss.backward()
                actor_optimiser.step()

                # Fit value function by regression on mean-squared error
                value_loss = ch.algorithms.a2c.state_value_loss(new_values,
                                                                returns)
                critic_optimiser.zero_grad()
                value_loss.backward()
                critic_optimiser.step()

            replay.empty()


if __name__ == '__main__':
    main()