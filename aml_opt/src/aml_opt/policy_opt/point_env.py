import numpy as np
import collections
import tensorflow as tf

class PointEnv(object):

    def __init__(self):
        self._step        = collections.namedtuple("Step", ["observation", "reward", "done", "info"])
        self._state_lim   = collections.namedtuple("State_Lim", ["high", "low"])
        self._action_lim  = collections.namedtuple("Action_Lim", ["high", "low"])
        self._state_low   = np.inf
        self._state_high  = np.inf
        self._action_low  = -0.1
        self._action_high = 0.1
        self._dim         = 2
        self._obs_dim     = 2
        self._action_dim  = 1 

    @property
    def observation_space(self):
        return self._state_lim(low=-np.inf, high=np.inf)

    @property
    def action_space(self):
        return self._action_lim(low=-0.1, high=0.1)

    def reset(self):
        self._state = np.random.uniform(-1, 1, size=(2,))
        observation = np.copy(self._state)
        return observation

    def step(self, action):

        self._state = self._state + action
        x, y = self._state
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = abs(x) < 0.01 and abs(y) < 0.01
        next_observation = np.copy(self._state)
        return self._step(observation=next_observation, reward=reward, done=done,  info=None)

    def render(self):
        print('current state:', self._state)

    def new_tensor_variable(self, name, extra_dims):
        
        return tf.placeholder(name=name,shape=(None, extra_dims+1),dtype=tf.float32)

    def spec(self):
        spec = {'observation_space': self.observation_space, 
                'action_space': self.action_space,}
        return spec
