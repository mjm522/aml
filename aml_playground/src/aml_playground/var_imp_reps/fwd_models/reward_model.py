import numpy as np
from collections import deque
from rl_algos.forward_models.gp_model import GPModel
from rl_algos.forward_models.reps_fwd_model import REPSFwdModel

class RewardModel(REPSFwdModel):

    def __init__(self, cost_fn, target, params, w_dim, num_data_points=30):

        self._cost_fn = cost_fn

        self._old_force = np.zeros(3)

        self._work_wt = params['work_weight']
        self._f_dot_wt = params['f_dot_weight']
        self._traj_wt = params['goal_weight']
        self._target = target

        self._w_dim = w_dim
        self._w_data  = deque(maxlen=num_data_points)
        self._reward_data = deque(maxlen=num_data_points)
        # self.cost = cost
        # self.context_model = context_model

        # self.reward_model = GPModel(x_dim=x_dim)


    def add_data(self, w, r):
        
        assert len(w) == self._w_dim
        self._w_data.append(w)
        self._reward_data.append(r)

    def fit(self):
        pass
        # S = np.asarray(self.context_model._data)
        # W = np.asarray(self._w_data)
        # R = np.asarray(self._reward_data)

        # if R.ndim == 1:
        #     R = R[:,None]                         
    
        # self.reward_model.fit(X=np.hstack([ S, W ]),  Y=R)


    # def predict(self, traj):

    #     penalty = self._cost_fn(traj)

    #     return penalty['total']

    def predict(self, s, w=None):

        return self._work_wt*np.multiply(s[6:9], s[3:6]) + self._traj_wt*np.linalg.norm(s[0:3]-self._target)