#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import random
import pdb

from trading_gym.utils.data.process import read_stock_history, index_to_date
from trading_gym.envs.portfolio_gym import PortfolioEnv
import cherry as ch
from models import ActorCritic



gamma=0.99
polyak=0.995
MAX_STEPS =1000
UPDATE_INTERVAL = 100
BATCH_SIZE = 32

def create_env_input():
    three_level_dir = os.path.abspath(os.path.join(__file__, "../../.."))
    data_dir = os.path.join(three_level_dir, "assets/dataset/stocks_history_target.h5")
    history, abbreviation = read_stock_history(data_dir)
    return history, abbreviation



# create env
history, abbreviation = create_env_input()
env = PortfolioEnv(history, abbreviation)
#env = ch.envs.Logger(env, interval=20)
env = ch.envs.Torch(env)


# create net
action_size = env.action_space.shape[0]
number_asset, seq_window, features_all = env.observation_space.shape
assert action_size == number_asset+1 
input_size = features_all-1

net = ActorCritic(input_size=input_size, hidden_size=50, action_size=action_size)
net_tgt = ActorCritic(input_size=input_size, hidden_size=50, action_size=action_size)
net_tgt.eval()
print(net_tgt)
net_tgt.load_state_dict(net.state_dict())


# create replay
replay = ch.ExperienceReplay()

# create loss function
criterion_mse = nn.MSELoss()

# create optimizer
optimizer_actor = torch.optim.Adam(net.actor.parameters(), lr=0.001)
optimizer_critic = torch.optim.Adam(net.critic.parameters(), lr=0.001)


def update(replay):
    # batch-data
    state_batch = replay.state()
    next_state_batch = replay.next_state()
    action_batch = replay.action()
    reward_batch = replay.reward()
    done_batch = replay.done()
    
    
    
    # Q-learning update
    action_state_value = net.apply_critic(state_batch, action_batch)
    action_state_value_tgt = net_tgt(next_state_batch)
    backup = reward_batch + gamma*(1-done_batch) * action_state_value_tgt
    q_loss = criterion_mse(action_state_value, backup)
    
    optimizer_critic.zero_grad()
    q_loss.backward()
    optimizer_critic.step()
    
    
    # Policy update
    action_state_value = net(state_batch)
    action_state_value_loss = -torch.mean(action_state_value)
    #pdb.set_trace()
    optimizer_actor.zero_grad()
    action_state_value_loss.backward()
    optimizer_actor.step()


state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

r,d,ep_ret, ep_len = 0, False,0, 0

update_times = 0

for step in range(1, MAX_STEPS + 1):
    
    if step == 732:
        pdb.set_trace()
    
    
    with torch.no_grad():
        action = net.get_action(state)
        action = action.squeeze(0)
        #pdb.set_trace()
        next_state, reward, done, _ = env.step(action)
        #pdb.set_trace()
        replay.append(state, action,reward, next_state, done)
        state = next_state
        if done:
            break
    
    #if step == 10:
    #    pdb.set_trace()
    
    if step % UPDATE_INTERVAL == 50:
        sample = random.sample(replay, BATCH_SIZE)
        batch = ch.ExperienceReplay(sample)
        update(batch)
        
        update_times +=1 
        print("update times:{}".format(update_times))

    
    