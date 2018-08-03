import numpy as np
from collections import deque
from rl_algos.forward_models.gp_model import GPModel
from rl_algos.forward_models.reps_fwd_model import REPSFwdModel

class RewardModel(REPSFwdModel):

    def __init__(self, w_dim, x_dim, cost, context_model, num_data_points=30):

        self._w_dim = w_dim
        self._w_data  = deque(maxlen=num_data_points)
        self._reward_data = deque(maxlen=num_data_points)
        self.cost = cost
        self.context_model = context_model

        self.reward_model = GPModel(x_dim=x_dim)


    def add_data(self, w, r):

        assert len(w) == self._w_dim
        self._w_data.append(w)
        self._reward_data.append(r)

    def fit(self):

        S = np.asarray(self.context_model._data)
        W = np.asarray(self._w_data)
        R = np.asarray(self._reward_data)

        if R.ndim == 1:
            R = R[:,None]                         
    
        self.reward_model.fit(X=np.hstack([ S, W ]),  Y=R)


    def predict(self, s, w):

        mu_r, sigma_r = self.reward_model.predict(np.hstack([ s, w ]))

        return mu_r