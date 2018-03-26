import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from aml_io.log_utils import aml_logging
from aml_io.io_tools import save_data, load_data

class PITrajOpt(object):

    """
    Class that utilizes PI2 algorithm to optimize a trajectory
    The algorithm, starts from a initial trajectory and incrementally modifies 
    it to find a lower cost trajectory
    """

    def __init__(self, config, cost_fn, visualize_fn=None):
        """
        Constructor of the PITraj class.
        Args: 
        config: 
                N: time steps of a trajectory
                K: no of rollouts that should be performed to collect trajectories
                h: the scalar tuning parameter of the PI2 algorithm
                gain: the coefficient that determines the amount of mixing
                max_iter: maximum number of iterations by the PI2 algorithm
                num_traj: total number of trajectories in the optimizer
                smooth_traj: whether to smooth a trajectory or not
                state_constraints: given as a dict {'min':[], 'max':[]}
                init_traj: optional input argument, gives an initial idea of the trajectory
                start: start of the trajectory, the optimizer optimises point except start and end
                goal: end point of the trajectory

        cost_fn: handle to the cost function, this function accepts and entire trajecotry sample (array[no_dimension, time_steps])
        and returns a scalar cost for the trajectory

        visualize_fn: handle to the plotting tool, this function accepts and an entire trajectory sample to display it
        """

        self._logger = aml_logging.get_logger(__name__)

        self._N        = config['timesteps']
        self._K        = config['no_rollouts']
        self._h        = config['h']
        self._gain     = config['gain']
        self._max_iter = config['max_iter']
        self._num_traj = len(config['start'])

        self._smooth_traj = config['smooth_traj']

        self._traj_file_name = None

        self._traj_min  = config['state_constraints']['min']
        self._traj_max  = config['state_constraints']['max']

        self._init_traj = np.zeros([self._N, self._num_traj])

        if  config['init_traj'] is None:
            for k in range(self._num_traj):
                self._init_traj[:, k] = np.linspace(config['start'][k], config['goal'][k], self._N)
        else:
            self._init_traj = config['init_traj']

        self._cost      = cost_fn

        self._visualize = visualize_fn

        if self._init_traj.ndim == 1:
            self._init_traj = self._init_traj[None, :]

        if self._init_traj.shape[0] != self._N:
            self._init_traj = self._init_traj.T

    
    def put_state_constraints(self, traj):
        """
        this function enforces the state constraints on the trajectory
        the exploration can lead to creation of trajectories that are outside the 
        constraints, this function de-limits them
        Args:
        traj: input trajectory
        """

        for k in range(self._num_traj):
            traj[:,k][traj[:,k] < self._traj_min[k]] = self._traj_min[k]
            traj[:,k][traj[:,k] > self._traj_max[k]] = self._traj_max[k]

        return traj


    def get_traj_samples(self, traj, gain=1e-0):
        """
        this function computes the trajectory samples for a given trajectory
        Args: 
        traj: a trajectory sample
        gain: exploratory gain. High value of this can make the algorithm numerically unstable.
        """

        traj_samples =  np.zeros([self._K, self._N, self._num_traj])
        cost_traj_samples = np.zeros([self._K, self._N])
        del_traj_samples = np.random.randn(self._K, self._N, self._num_traj)

        for k in range(self._K):
            
            traj_samples[k,:,:] = self.put_state_constraints( (traj[None, :, :] + gain*del_traj_samples[k,:,:]).squeeze() )
            
            cost_traj_samples[k,:] = self._cost(traj_samples[k,:,:])

        return traj_samples, del_traj_samples, cost_traj_samples

    def compute_traj_change(self, cost_samples, del_traj_samples):
        """
        this function compute the required trajectory change based on cost of each trajectory
        low cost trajectories will be preferred to high cost trajectories
        Args: 
        cost_samples: cost values assosciated with each trajectory rollout
        del_traj_samples: the exploratory trajectory change that was supplied 
        """
        exp_cost_samples = np.exp(-self._h*cost_samples)
        denominators = np.sum(exp_cost_samples, axis=0)
        del_traj_samples = del_traj_samples/denominators[None,:,None]
        del_traj_samples = np.multiply(exp_cost_samples[:, :, None], del_traj_samples)

        return np.sum(del_traj_samples, axis=0)

    def savitsky_gollay_filter(self, traj):
        """
        this is a smoothing filter that helps to make 
        the exploratory trajectories smooth.
        This is an optional part and is implemented depending on the self._smooth_traj variable
        Args:
        traj: input trajectory
        """
        return savgol_filter(x=traj, window_length=5, polyorder=2)

    def modify_traj(self, traj, traj_change_gain=1e-1):
        """
        this function computes the stochastic incremental change to the existing path
        depending on the forward rollouts
        Args:
        traj: input trajectory
        traj_change_gain: the parameter that scales the change to a trajectory, this should not be a very large value
        since it can make the convergence numerically unstable
        """

        traj_samples, del_traj_samples, cost_traj_samples = self.get_traj_samples(traj)

        traj_change = self.compute_traj_change(cost_traj_samples, del_traj_samples)

        tmp_traj =  self.put_state_constraints(  (traj + traj_change_gain*traj_change) )[1:self._N-1, :]

        for k in range(self._num_traj):
            if self._smooth_traj:
                traj[1:self._N-1, k] = self.savitsky_gollay_filter(tmp_traj[:,k])
            else:
                traj[1:self._N-1, k] = tmp_traj[:,k]

        cost_traj = self._cost(traj)

        if self._visualize is not None:
            self._visualize(traj)
        
        return traj, np.sum(cost_traj)

    def run(self):
        """
        the main function that runs the algorithm
        Args: None
        """

        traj_final =  self._init_traj.copy()
        
        for t in range(self._max_iter):

            traj_final, cost = self.modify_traj(traj_final)

            print "Iteration %d and cost is %f"%(t, cost)
        
        if self._traj_file_name is not None:
            save_data(traj_final, self._traj_file_name)
        
        self._logger.info("completed iterations")
        
        return traj_final
        


