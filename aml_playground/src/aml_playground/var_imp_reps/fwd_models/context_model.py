import numpy as np
from collections import deque
from rl_algos.forward_models.reps_fwd_model import REPSFwdModel

np.random.seed(123)

class ContextModel(REPSFwdModel):

    def __init__(self, spring_base, req_traj, spring_k, min_vel, max_vel, context_dim, num_data_points=30):

        self._spring_base = spring_base
        self._req_traj = req_traj
        self._spring_k = spring_k
        self._traj_start = self._req_traj[0,:]
        self._traj_end =  self._req_traj[-1,:]
        self._min_vel = min_vel
        self._max_vel = max_vel
        self._data = deque(maxlen=num_data_points)
        self._context_dim = context_dim

    def add_data(self, datum):

        assert len(datum) == self._context_dim
        self._data.append(datum)
    

    def fit(self, X=None, Y=None):
        pass

        # S = np.asarray(self._data)

        # self._mean = np.mean(S, axis=0)
        # self._std  = np.std(S, axis=0)


    def sample(self, x_noise=np.array([0.01, 0.01, 0]), x_dot_noise=np.array([0.001, 0.001, 0.001])):

        x_sample = np.zeros(3)
        x_dot_sample = np.zeros(3)

        for k in range(3):
            x_sample[k] = np.random.uniform(self._traj_start[k], self._traj_end[k]) + x_noise[k]*np.random.randn(1)
            x_dot_sample[k] = np.random.uniform(self._min_vel[k], self._max_vel[k]) + x_dot_noise[k]*np.random.randn(1)

        f_sample = self._spring_k*(self._spring_base-x_sample)

        return np.hstack([x_sample, x_dot_sample, f_sample])

        # if isinstance(self._mean, np.float):
        
        #     return np.asarray([np.random.normal(loc=self._mean, scale=self._std)])

        # else:
        
        # return np.random.multivariate_normal(mean=self._mean, cov=np.outer(self._std, self._std))

