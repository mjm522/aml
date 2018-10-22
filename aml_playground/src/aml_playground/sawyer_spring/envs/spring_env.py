
import numpy as np

class SpringEnv():

    def __init__(self, string_offset, random_state, k=2.5):

        self._x = 0
        self._stiffness = k
        self._force =  self._stiffness*self._x
        self._random_state = random_state
        self._string_offset = string_offset

    def compute_force(self, ee_x):
        
        if self._x > self._string_offset:
            self._x = ee_x-self._string_offset
            self._force = self._stiffness*self._x 
        else:
            self._force = 0.

        return self._force


