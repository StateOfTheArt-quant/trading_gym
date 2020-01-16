import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class Actor(nn.Module):
    
    def __init__(self, input_size, hidden_size, action_size,stochastic=True):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size*action_size, out_features=action_size)
        self.softmax = nn.Softmax(dim=1)
        
        if stochastic:
            self.policy_log_std = nn.Parameter(torch.zeros(action_size))       
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, x.shape[-2], x.shape[-1])
        x, _ = self.lstm(x)
        last_vector = x[:,-1,:]
        x = last_vector.view(batch_size,-1)
        output = self.fc(x)
        output = self.softmax(output)
        return output

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size*action_size + action_size, out_features=1)
    
    def forward(self, x, action):
        batch_size = x.shape[0]
        x = x.view(-1, x.shape[-2], x.shape[-1])
        x, _ = self.lstm(x)
        last_vector = x[:,-1,:]
        x = last_vector.view(batch_size,-1)
        x = torch.cat([x, action], dim=1)
        x = self.fc(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, action_size, stochatic=True):
        super(ActorCritic, self).__init__()
        self.actor = Actor(input_size, hidden_size, action_size, stochatic)
        self.critic = Critic(input_size, hidden_size, action_size)
    
    def forward(self, state):
        cov_mat = torch.diag(self.actor.policy_log_std.exp())
        policy = MultivariateNormal(self.actor(state), cov_mat)
        action = policy.sample()
        log_prob = policy.log_prob(action)
        value = self.critic(state, action)
        return action, {
                        'mass': policy,
                        'log_prob': log_prob,
                        'value': value}

       
if __name__ == "__main__":
    sample_size = 2
    num_stocks = 5
    seq_window=6
    feature = 4
    
    x = torch.randn((sample_size, num_stocks, seq_window, feature))
    
    actor = Actor(input_size=feature, hidden_size=50,action_size=5)
    critic = Critic(input_size=feature, hidden_size=50,action_size=5)
    
    action = actor(x)    
    state_action_value = critic(x,action) 
    