#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import random
import pdb

import cherry as ch
from trading_gym.utils.data.toy import create_toy_data
from trading_gym.envs import PortfolioTradingGym
from trading_gym.wrapper import Numpy
from models import Actor, Critic


def create_target_network(network):
    target_network = copy.deepcopy(network)
    for param in target_network.parameters():
        param.requires_grad = False
    return target_network

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return torch.tensor(x, dtype=torch.float)


DISCOUNT = 0.99
LEARNING_RATE_ACTOR = 0.0005
LEARNING_RATE_CRITIC =0.001
MAX_STEPS = 100000
BATCH_SIZE = 32
REPLAY_SIZE = 50000
UPDATE_INTERVAL = 20
UPDATE_START = 20000
POLYAK_FACTOR = 0.995

SEED = 111
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
    
    actor = Actor(input_size=input_size, hidden_size=50, action_size=action_size)
    critic = Critic(input_size=input_size, hidden_size=50, action_size=action_size)
    
    target_actor = create_target_network(actor)
    target_critic = create_target_network(critic)
    actor_optimiser = optim.Adam(actor.parameters(), lr=LEARNING_RATE_ACTOR)
    critic_optimiser = optim.Adam(critic.parameters(), lr=LEARNING_RATE_CRITIC)
    replay = ch.ExperienceReplay()
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(action_size))
    
    
    def get_action(state):
        action = actor(state)
        action = action +  ou_noise()[0]
        return action      

    def get_random_action(state):
        action = torch.softmax(torch.randn(action_size), dim=0)
        return action


    for step in range(1, MAX_STEPS + 1):
        with torch.no_grad():
            
            if step < UPDATE_START:
                replay += env.run(get_random_action, steps=1)
            else:
                replay += env.run(get_action, steps=1)

        replay = replay[-REPLAY_SIZE:]
        if step > UPDATE_START and step % UPDATE_INTERVAL == 0:
            sample = random.sample(replay, BATCH_SIZE)
            batch = ch.ExperienceReplay(sample)

            next_values = target_critic(batch.next_state(),
                                        target_actor(batch.next_state())
                                        ).view(-1, 1)
            values = critic(batch.state(), batch.action()).view(-1, 1)            
            rewards = ch.normalize(batch.reward())
            #rewards = batch.reward()/100.0   change the convergency a lot
            value_loss = ch.algorithms.ddpg.state_value_loss(values,
                                                             next_values.detach(),
                                                             rewards,
                                                             batch.done(),
                                                             DISCOUNT)
            critic_optimiser.zero_grad()
            value_loss.backward()
            critic_optimiser.step()

            # Update policy by one step of gradient ascent
            policy_loss = -critic(batch.state(), actor(batch.state())).mean()
            actor_optimiser.zero_grad()
            policy_loss.backward()
            actor_optimiser.step()

            # Update target networks
            ch.models.polyak_average(target_critic,
                                     critic,
                                     POLYAK_FACTOR)
            ch.models.polyak_average(target_actor,
                                     actor,
                                     POLYAK_FACTOR)

if __name__ == '__main__':
    main()