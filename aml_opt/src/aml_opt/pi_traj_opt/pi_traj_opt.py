import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from aml_io.io_tools import save_data, load_data

np.random.seed(42)

class PITrajOpt(object):

    def __init__(self, config, cost_fn, visualize_fn=None):
        
        self._N        = config['timesteps']
        self._K        = config['no_rollouts']
        self._h        = config['h']
        self._gain     = config['gain']
        self._max_iter = config['max_iter']
        self._num_traj = len(config['start'])

        self._traj_file_name = None

        self._traj_min  = config['state_constraints']['min']
        self._traj_max  = config['state_constraints']['max']

        if  config['init_traj'] is None:
            _init_traj = []
            for k in range(self._num_traj):
                _init_traj.append(np.linspace(config['start'][k], config['goal'][k], self._N))
            self._init_traj = np.asarray(_init_traj)
        else:
            self._init_traj = config['init_traj']

        self._cost      = cost_fn

        self._visualize = visualize_fn

        if self._init_traj.ndim == 1:
            self._init_traj = self._init_traj[None, :]

        self._init_traj = self._init_traj.T

    
    def put_state_constraints(self, traj):

        for k in range(self._num_traj):
            traj[:,k][traj[:,k]<self._traj_min[k]] = self._traj_min[k]
            traj[:,k][traj[:,k]>self._traj_max[k]] = self._traj_max[k]

        return traj


    def get_traj_samples(self, traj, gain=1e-0):

        traj_samples =  np.zeros([self._K, self._N, self._num_traj])
        cost_traj_samples = np.zeros([self._K, self._N])
        del_traj_samples = np.random.randn(self._K, self._N, self._num_traj)

        for k in range(self._K):
            
            traj_samples[k,:,:] = self.put_state_constraints( (traj[None, :, :] + gain*del_traj_samples[k,:,:]))

            cost_traj_samples[k,:] = self._cost(traj_samples[k,:,:])

        return traj_samples, del_traj_samples, cost_traj_samples

    def compute_traj_change(self, cost_samples, del_traj_samples):
        exp_cost_samples = np.exp(-self._h*cost_samples)
        denominators = np.sum(exp_cost_samples, axis=0)
        del_traj_samples = del_traj_samples/denominators[None,:,None]
        del_traj_samples = np.multiply(exp_cost_samples[:, :, None], del_traj_samples)

        return np.sum(del_traj_samples, axis=0)

    def savitsky_gollay_filter(self, traj):
        return savgol_filter(x=traj, window_length=5, polyorder=2)

    def modify_traj(self, traj):

        traj_samples, del_traj_samples, cost_traj_samples = self.get_traj_samples(traj)

        traj_change = self.compute_traj_change(cost_traj_samples, del_traj_samples)

        tmp_traj =  self.put_state_constraints(  (traj + 1e-1*traj_change) )[1:self._N-1, :]

        for k in range(self._num_traj):
            traj[1:self._N-1, k] = self.savitsky_gollay_filter(tmp_traj[:,k])

        cost_traj = self._cost(traj)

        if self._visualize is not None:
            self._visualize(traj)
        
        return traj, np.sum(cost_traj)

    def run(self):

        traj_final =  self._init_traj.copy()
        
        for t in range(self._max_iter):

            traj_final, cost = self.modify_traj(traj_final)

            print "Iteration %d and cost is %f"%(t, cost)
        
        if self._traj_file_name is not None:
            save_data(traj_final, self._traj_file_name)
        
        raw_input("Waiting...")


