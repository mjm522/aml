import numpy as np
import collections


class PointEnv(object):

    def __init__(self, config):
        self._data   = collections.namedtuple("data", ["state", "action", "observation", "reward"])
        self._state_dim  = 2
        self._action_dim = 1
        self._dt = config['dt']
        self._A  = np.array([[1.,self._dt],[0., 1.]])
        self._B  = np.array([[0.],[self._dt]])
        self._start_state = np.array([3., 0.])
        self._goal  = np.array([0., 0.])


    def reset(self):
        return np.random.randn(self._state_dim)


    def step(self, state, action):
        '''
        step function to find the next state given a state and action
        '''

        new_state = np.dot(self._A, state) + np.dot(self._B, action)

        reward = -np.linalg.norm(new_state-self._goal)

        return self._data(state=state, action=action, observation=new_state, reward=reward)