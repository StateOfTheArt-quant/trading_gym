# reference
# https://github.com/implementation-matters/code-for-paper/blob/master/src/policy_gradients/torch_utils.py
import gym
import numpy as np


class RunningStat(object):
    '''
    Keeps track of first and second moments (mean and variance)
    of a streaming time series.
     Taken from https://github.com/joschu/modular_rl
     Math in http://www.johndcook.com/blog/standard_deviation/
    '''
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)
    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._M
    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)
    @property
    def std(self):
        return np.sqrt(self.var)
    @property
    def shape(self):
        return self._M.shape

class Identity:
    '''
    A convenience class which simply implements __call__
    as the identity function
    '''
    def __call__(self, x, *args, **kwargs):
        return x

    def reset(self):
        pass


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """
    def __init__(self, prev_filter, shape, center=True, scale=True, clip=None, eps=1e-8):
        assert shape is not None
        self.center = center
        self.scale = scale
        self.clip = clip
        self.eps = eps
        self.rs = RunningStat(shape)
        self.prev_filter = prev_filter

    def __call__(self, x, **kwargs):
        x = self.prev_filter(x, **kwargs)
        self.rs.push(x)
        if self.center:
            x = x - self.rs.mean
        if self.scale:
            if self.center:
                x = x / (self.rs.std + self.eps)
            else:
                diff = x - self.rs.mean
                diff = diff/(self.rs.std + self.eps)
                x = diff + self.rs.mean
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def reset(self):
        self.prev_filter.reset()

class RewardFilter:
    """
    Incorrect reward normalization [copied from OAI code]
    update return
    divide reward by std(return) without subtracting and adding back mean
    """
    def __init__(self, prev_filter, shape, gamma, clip=None, eps=1e-8):
        assert shape is not None
        self.gamma = gamma
        self.eps = eps
        self.prev_filter = prev_filter
        self.rs = RunningStat(shape)
        self.ret = np.zeros(shape)
        self.clip = clip

    def __call__(self, x, **kwargs):
        x = self.prev_filter(x, **kwargs)
        self.ret = self.ret * self.gamma + x
        self.rs.push(self.ret)
        x = x / (self.rs.std + self.eps)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x
    
    def reset(self):
        self.ret = np.zeros_like(self.ret)
        self.prev_filter.reset()


class Normalizer(gym.Wrapper):
    """
    Normalizes the states and rewards with a running average.
    **Arguments**
     * **env** (Environment) - Environment to normalize.
     * **norm_states** (bool, *optional*, default=True) - Whether to normalize the
       states.
     * **norm_rewards** (str, *optional*, default="returns") - Whether to normalize the
       rewards or returns.
     * **clip_states** (float, *optional*, default=None) - Clip each state
       dimension between [-clip_states, clip_states].
     * **clip_rewards** (float, *optional*, default=None) - Clip rewards
       between [-clip_rewards, clip_rewards].
     * **gamma** (float, *optional*, default=0.99) - Discount factor for
       rewards running averages.
     * **eps** (float, *optional*, default=1e-8) - Numerical stability.
     """
    def __init__(self, env, norm_states=True, norm_rewards="returns", clip_obs=None, clip_rew=None, gamma=0.99, eps=1e-8):
        super(Normalizer, self).__init__(env)
        self.env = env
        clip_obs = None if clip_obs < 0 else clip_obs
        clip_rew = None if clip_rew < 0 else clip_rew
        
        # Support for state normalization
        self.state_filter = Identity()
        if norm_states:
            self.state_filter = ZFilter(self.state_filter, shape=env.observation_space.shape, clip=clip_obs)
        
        # Support for reward normalization
        self.reward_filter = Identity()
        if norm_rewards == "rewards":
            self.reward_filter = ZFilter(self.reward_filter, shape=(), center=False, clip=clip_rew)
        elif norm_rewards == "returns":
            self.reward_filter = RewardFilter(self.reward_filter, shape=(), gamma=gamma, clip=clip_rew)
        
        self.total_true_reward = 0.0
    
    def reset(self):
        """the state_filter and return_filter actually never reset,
        to keep track running stats all over the episodes rather than one[i have a little bit concern]"""
        start_state = self.env.reset()
        self.total_true_reward = 0.0
        self.counter = 0.0
        self.state_filter.reset()
        return self.state_filter(start_state)
    
    def step(self, action):
        state, reward, is_done, info = self.env.step(action)
        state = self.state_filter(state)
        self.total_true_reward += reward
        self.counter += 1
        _reward = self.reward_filter(reward)
        if is_done:
            info['done'] = (self.counter, self.total_true_reward)
        return state, _reward, is_done, info
    
