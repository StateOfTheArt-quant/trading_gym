#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pdb

class Actor(nn.Module):
    
    def __init__(self, input_size, hidden_size, action_size):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        
        self.fc = nn.Linear(in_features=hidden_size*action_size, out_features=action_size)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        batch_size = x.shape[0]

        x = x.view(-1,x.shape[-2],x.shape[-1])
        x, _ = self.lstm(x)
        last_vector = x[:,-1,:]
        x = last_vector.view(batch_size,-1)
        output = self.fc(x)
        output = self.softmax(output)
        return output.squeeze(0)

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        
        self.fc = nn.Linear(in_features=hidden_size*action_size + action_size, out_features=1)
    
    def forward(self, x, action):
        batch_size = x.shape[0]
        x = x.view(-1, x.shape[-2],x.shape[-1])
        x, _ = self.lstm(x)
        last_vector = x[:,-1,:]
        x = last_vector.view(batch_size,-1)
        #pdb.set_trace()
        x = torch.cat([x, action],dim=1)
        x = self.fc(x)
        return x
    

class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super(ActorCritic, self).__init__()
        self.actor = Actor(input_size, hidden_size, action_size)
        self.critic = Critic(input_size, hidden_size, action_size)
    
    def forward(self, x):
        pi = self.actor(x)
        q_pi = self.apply_critic(x, pi)
        return q_pi
    
    def apply_critic(self, x, action):
        q =  self.critic(x,action)
        return q
    
    def get_action(self, x):
        with torch.no_grad():
            action = self.actor(x)
        return action
        
if __name__ == "__main__":
    sample_size = 2
    num_stocks = 5
    seq_window=6
    feature = 4
    
    x = torch.randn((sample_size,num_stocks,seq_window, feature))
    
    actor = Actor(input_size=feature, hidden_size=50,action_size=5)
    critic = Critic(input_size=feature, hidden_size=50,action_size=5)
    
    action = actor(x)
    
    state_action_value = critic(x,action) 
    
    