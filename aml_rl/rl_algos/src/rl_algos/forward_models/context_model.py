import numpy as np
from collections import deque
from rl_algos.forward_models.reps_fwd_model import REPSFwdModel


class ContextModel(REPSFwdModel):

    def __init__(self, context_dim, num_data_points=30):

        self._context_dim = context_dim
        self._data = deque(maxlen=num_data_points)
        self._mean = None
        self._std  = None

    def add_data(self, datum):

        assert len(datum) == self._context_dim
        self._data.append(datum)
    

    def fit(self, X=None, Y=None):

        S = np.asarray(self._data)

        self._mean = np.mean(S, axis=0)
        self._std  = np.std(S, axis=0)


    def sample(self):

        # if isinstance(self._mean, np.float):
        
        #     return np.asarray([np.random.normal(loc=self._mean, scale=self._std)])

        # else:
        
        return np.random.multivariate_normal(mean=self._mean, cov=np.outer(self._std, self._std))