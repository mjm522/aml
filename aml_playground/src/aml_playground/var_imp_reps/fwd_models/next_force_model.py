import numpy as np

class NextForcePredictModel():

    def __init__(self, spring_k=2):

        self._K = spring_k

    def add_data(self, traj):
        return

    def fit(self, X=None, Y=None):
        pass

    def predict_force(self, spring_mean, x_t):
        """
        send in next state
        """

        return self._K*(spring_mean-x_t)