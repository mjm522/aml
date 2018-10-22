import time
import numpy as np
from aml_playground.sawyer_spring.envs.spring_env import SpringEnv

class EEEnv():

    def __init__(self, goal, random_state, string_offset=2., k=2.5):
        
        self._ee_x = 0.

        self._ee_dx = 0.

        self._t_old = time.time()

        self._goal = goal
        
        self._spring_env =  SpringEnv(string_offset=string_offset, random_state=random_state, k=k)
        
        self._random_state = random_state


    def context(self):

        self._s = self._spring_env._force

        return np.array([self._s])

    def force_sensor(self):
    
        return self._spring_env.compute_force(self._ee_x)

    def reward(self, w, s):
        
        x_offset = np.array([self._goal-self._ee_x]) + s.dot(np.array([[0.2]])).dot(s) + 0.01*w.dot(w)

        return -np.array([x_offset.dot(x_offset)])

    def execute_policy(self, w, s, **kwargs):

        delta_t = time.time()-self._t_old

        f=w[0]*(self._goal-self._ee_x) - w[1]*self._ee_dx - self._spring_env._force
        self._ee_dx += f*delta_t
        self._ee_x += self._ee_dx*delta_t
        self._t_old += delta_t

        return None, self.reward(w, s)

    def _reset(self):

        self._ee_x = 0.
        self._ee_dx = 0.
        self._t_old = time.time()
