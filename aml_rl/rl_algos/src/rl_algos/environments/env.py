import numpy as np

class Env():

    def __init__(self, x0, n_samples_per_update, random_state):
        
        self._x0 = x0
        
        self._random_state = random_state

        self.n_samples_per_update = n_samples_per_update

    def context(self):

        context = self._random_state.rand(1)* 10.0 - 5.0

        return context


    def reward(self, x, s):
        
        x_offset = x + s.dot(np.array([[0.2]])).dot(s)

        return -np.array([x_offset.dot(x_offset)])

    def execute_policy(self, w, s):
        
        return None, self.reward(w, s)




