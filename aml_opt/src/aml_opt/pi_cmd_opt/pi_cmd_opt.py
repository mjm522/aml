import time
import numpy as np
from aml_io.log_utils import aml_logging

class PICmdOpt(object):
    """
    Bare bone implementation of the PI2 algorithm
    it is written in a way as to optimize a set of control commands
    of a given length
    """

    def __init__(self, config):
        """
        Constructor of the class
        Arguments:
                config :   (dict type)
                        Arguments:
                                'max_iterations': maximum iterations for the PI2
                                'dt': time step 
                                'N': total length of the trajectory
                                'h': weighting of the exponential map
                                'rho': variance of the exploration noise
                                'K': gain of the cost
                                'cmd_dim': number of dimensions of the commands
                                'state_dim': number of dimensions of the states
                                'random_seed': seed for the random generator
                                'init_policy': initial policy type, pass 'init_random' for random policy
                                'ctrl_constraints':control constraints, pass None if not needed
        """

        np.random.seed(config['random_seed'])

        self._logger = aml_logging.get_logger(__name__)

        self._dt = config['dt']
        self._K  = config['K']      
        self._N  = config['N']

        self._max_itr = config['max_iterations']

        self._rho = config['rho'] # i.e np.sqrt(1./rho)

        if isinstance(self._rho, float):
            self._rho = np.ones(self._N)*self._rho
            self._rho_bkup = self._rho.copy()

        self._h   = config['h']

        if config['ctrl_constraints'] is not None:

            self._cmd_min  = config['ctrl_constraints']['min']
            self._cmd_max  = config['ctrl_constraints']['max']
        else:
            self._cmd_min = None
            self._cmd_max = None

        self._cmd_dim   = config['cmd_dim']
        self._state_dim = config['state_dim']

        self._x0 = np.zeros(self._state_dim)

    def init(self, x0):
        """
        this function restates after each control action is taken
        in the model predictive fashion
        Args:
        x0 = starting state of each iteration
        """

        self._x0 = x0

    def put_control_constraints(self, u_list):

        if self._cmd_min is None or self._cmd_max is None:
            return u_list
        """
        this function enforces the state constraints on the trajectory
        the exploration can lead to creation of trajectories that are outside the 
        constraints, this function de-limits them
        Args:
        u_list: input control trajectory
        """

        for k in range(self._cmd_dim):
            u_list[:,k][u_list[:,k] < self._cmd_min[k]] = self._cmd_min[k]
            u_list[:,k][u_list[:,k] > self._cmd_max[k]] = self._cmd_max[k]

        return u_list

    # Random variation of control command (a random change in the control command)
    def calc_delu(self, t):
        """
        compute the delta u increament exploration that needs to be performed
        Args:
        t = time index
        """
        return self._rho[t]*np.random.randn(self._cmd_dim) #/np.sqrt(self._dt))


    def compute_du_list(self):

        return np.multiply(self._rho, np.random.randn(self._cmd_dim, self._N))


    def forward_samples(self, us):
        """
        compute and collect the forward samples
        Args: 
        us : set of control commands
        """
        
        delus_samples = np.zeros((self._K, self._N, self._cmd_dim))
        xs_samples = np.zeros((self._K, self._N, self._state_dim)) # Forward trajectory samples
        ss = np.zeros((self._K,self._N,)) # Costs
    
        # delus_samples = self.calc_delu() # del u, variations of control commands
        # This can be paralised with a thread for every sample k
        for k in range(0, self._K):

            xs_samples[k, : , :], ss[k,:], delus_samples[k, :, :] = self.forward_rollout(us, k)

        return xs_samples, ss, delus_samples

    def compute_traj_probability(self, ss):
        """
        Compute the goodness of trajectory and normalise it
        Basically it means we are finding which trajectories are more probable.
        Args: ss = trajectory costs for all samples
        """
        exp_ss = np.exp(-self._h*ss)
        denominators = np.sum(exp_ss, axis=0)
        exp_ss =  exp_ss[:,:,None]/denominators[None,:,None]
        return exp_ss


    def compute_u_change(self, traj_prob, delus_samples):
        """
        Compute the incremental u_change of iterative update
        Args: traj_prob = weights of individual trajectories
            delus_samples = the exploation that was added to the control commands
        """
        
        delus_samples = np.multiply(traj_prob, delus_samples)
        delus_samples[np.isnan(delus_samples)] = 0.

        return np.sum(delus_samples, axis=0).T

    def run(self, u_list):
        """
        this is function that could be parrallelized to perform parralled rollouts of several
        rollouts for a given set of u_list by perturbing this same list in different amounts
        Currently this funciton is not parrallelized and is performs on a series
        Args: u_list = list of control commands
        """
        for k in range(self._max_itr):

            self._logger.debug("Iteration \t"); self._logger.debug(k)

            tic = time.time()

            xs_samples, ss, delus_samples = self.forward_samples(u_list)

            traj_prob = self.compute_traj_probability(ss)

            self._old_samples = delus_samples
            self._old_costs   = np.sum(ss, axis=1)
            
            toc = time.time() - tic

            delta_u = self.compute_u_change(traj_prob, delus_samples)

            # print "Elapsed time \t", toc

            u_list = u_list + delta_u

        return u_list, xs_samples, traj_prob, delus_samples
