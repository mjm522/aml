import time
import random
import numpy as np

class MPPIController(object):
    """
    Model predictive path integral controller
    The class is able to choose between various types of dynamics models,
    perform forward roll-outs and obtain optimal actions.
    The actions are performed in a model predictive fashion.
    For more details on the approach refer: http://ieeexplore.ieee.org/iel7/8215882/8239529/08246918.pdf
    """

    def __init__(self, fdyn, cost, config):
        """
        Constructor of the class
        Args:
        fdyn = dynamics function handle (this function will be used in the forward simulation)
        cost = cost function handle (this function computes the cost or weight or trajectory, 
               the types input args of this function can be seen aml_planner/cost)
        config file:
                   dynamics_type = whether is the passed dynamics model is a learned dynamics model
                   dt = time step of forward simulation (type: float)
                   K  = number of look ahead steps of the model predictive approach(type: int)
                   N  = number of samples (type: int)
                   rho = exploation term, this reduces over time on each model predictive iteration (type: float)
                   cmd_dim = dimensions of the control input (type: int)
                   state_dim = dimensions of state control (type: int)
                   use_mc_unc_estimation = whether to use uncertainty (type : bool)
                   resample_proportion = to perform importantance based resampling of the trajectories (optional, type: float)
        """

        np.random.seed(config['random_seed'])

        self._is_learnt_dyn = config['dynamics_type'] == 'learnt_dyn'

        self._dt = config['dt']
        self._K  = config['K']      
        self._N  = config['N'] 

        self._rho = config['rho'] # i.e np.sqrt(1./rho)

        if isinstance(self._rho, float):
            self._rho = np.ones(self._N)*self._rho
            self._rho_bkup = self._rho.copy()

        self._h   = config['h']

        self._cmd_dim   = config['cmd_dim']
        self._state_dim = config['state_dim']

        self._use_mc_unc = config['use_mc_unc_estimation']

        if 'resample_proportion' in config.keys():
            self._resamp_p = config['resample_proportion']
        else:
            self._resamp_p = 0.5

        self._x0 = np.zeros(self._state_dim)

        self._fdyn = fdyn

        self._cost = cost

        self._c = 0

        self._old_samples = None
        self._old_costs = None
        self._resamples = None

        self._itr = 0

        self._conv_thresh = 0.1
        self._max_itr_conv = 20


    def init(self, x0):
        """
        this function restates after each control action is taken
        in the model predictive fashion
        Args:
        x0 = starting state of each iteration
        """

        self._x0 = x0


    # Random variation of control command (a random change in the control command)
    def calc_delu(self, t):
        """
        compute the delta u increament exploration that needs to be performed
        Args:
        t = time index
        """
        
        return self._rho[t]*np.random.randn(self._cmd_dim) #/np.sqrt(self._dt))

        # return self._rho*np.random.randn(self._K, self._N, self._cmd_dim)/np.sqrt(self._dt)

    # Computes average uncertainty by predicting small random perturbations on x_prev, u
    def get_unc(self, x_prev, u, dt, n, noise = 0.001):
        """
        compute the uncertainty of a rollout using monte-carlo sampling
        Args: 
        x_prev = last state
        u = control command
        dt = time step
        n = number of samples
        noise = peturbation
        """

        if not self._is_learnt_dyn:
            return 0.0

        sigmas = np.zeros(n)
        for i in range(n):
            noisy_u = u + np.random.randn(*u.shape)*noise
            noisy_x = x_prev + np.random.randn(*x_prev.shape)*noise
            _, tmp, _ = self._fdyn(noisy_x, noisy_u)

            sigmas[i] = np.mean(np.sqrt(tmp))

        return np.mean(sigmas)

    def forward_rollout(self, u_list, k):
        """
        perform a single rollout for a set of control commands
        Args:
        u_list = list of control commands to perform rollout
        k = number of resampling to be done if required
        """

        xs_samples = np.zeros((self._N, self._state_dim)) # Forward trajectory samples
        ss = np.zeros(self._N) # Costs

        if (self._resamples is not None) and (k < self._resamples.shape[0]):
            delus_samples = self._resamples[k,:,:]
            resample_flag = True
        else:
            delus_samples = np.zeros((self._N, self._cmd_dim))
            resample_flag = False
        
        xs_samples[0,:] = self._x0.copy()

        def comp_unc(x, us, dt, sigma, iter):

            if self._use_mc_unc:
                return self.get_unc(x_prev, u_list[:,t] + du, self._dt,7)
            else:
                return np.sum(np.sqrt(sigma))/5.0


        for t in range(0,self._N):
            x_prev = xs_samples[t,:]
            
            if resample_flag:
                du = delus_samples[t]
            else:
                du = self.calc_delu(t)
            
            x_next, sigma, _  = self._fdyn(x_prev, u_list[:,t] + du)
            sigma = comp_unc(x_prev, u_list[:,t] + du, self._dt,sigma,7)

            xs_samples[t,:] = x_next
            ss[t] += ss[t] + self._cost(x_next, u_list[:,t], du, sigma, t+1)
            if ss[t] < 1e-4 or np.isnan(ss[t]):
                ss[t] = 0.
            delus_samples[t,:] = du

        return xs_samples, ss, delus_samples


    def forward_samples(self, us):
        """
        compute and collect the forward samples
        Args: 
        us : set of control commands
        """

        if (self._itr != 0): # and (self._resamp_p > 0)
            self._resamples = self.resample_samples()
        
        delus_samples = np.zeros((self._K, self._N, self._cmd_dim))
        xs_samples = np.zeros((self._K, self._N, self._state_dim)) # Forward trajectory samples
        ss = np.zeros((self._K,self._N,)) # Costs
    
        # delus_samples = self.calc_delu() # del u, variations of control commands
        # This can be paralised with a thread for every sample k
        for k in range(0, self._K):

            xs_samples[k, : , :], ss[k,:], delus_samples[k, :, :] = self.forward_rollout(us, k)

        return xs_samples, ss, delus_samples


    def resample_samples(self):
        """
        The function that performs sampling importance resampling
        """

        resamples = []
        #proportion of old samples: p
        no_resamples = int(self._K*self._resamp_p)

        #resampling with replacement
        index = int(random.random()*no_resamples)
        beta = 0.
        max_weight = np.max(self._old_costs)
        for i in range(no_resamples):
            beta += random.random()*2.*max_weight

            while beta > self._old_costs[index]:
                beta -= self._old_costs[index]
                index = (index+1) % self._N
            resamples.append(self._old_samples[index,:,:])

        return np.asarray(resamples)

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


    # def compute_u_change(self, ss, delus_samples):

    #     # Offset trick (see: http://cs231n.github.io/linear-classify/#softmax)
    #     # max_s = np.max(ss)
    #     # ss -= max_s

    #     exp_ss = np.exp(-self._h*ss)

    #     denominators = np.sum(exp_ss, axis=0)

    #     delus_samples = delus_samples/denominators[None,:,None]

    #     delus_samples = np.multiply(exp_ss[:, :, None], delus_samples)

    #     delus_samples[np.isnan(delus_samples)] = 0.

    #     return np.sum(delus_samples, axis=0).T 


    def fun_to_parrallelize(self, u_list):
        """
        this is function that could be parrallelized to perform parralled rollouts of several
        rollouts for a given set of u_list by perturbing this same list in different amounts
        Currently this funciton is not parrallelized and is performs on a series
        Args: u_list = list of control commands
        """
        tic = time.time()

        xs_samples, ss, delus_samples = self.forward_samples(u_list)

        traj_prob = self.compute_traj_probability(ss)

        self._old_samples = delus_samples
        self._old_costs   = np.sum(ss, axis=1)
        
        toc = time.time() - tic

        delta_u = self.compute_u_change(traj_prob, delus_samples)

        # print "Elapsed time \t", toc

        u_list = u_list + delta_u

        self._c += 1

        self._cost._c = self._c

        return u_list, xs_samples, traj_prob, delus_samples


    def per_step_iteration(self, u_list):
        """
        this is the main loop that performs iterations to find a locally
        optimal control input to be passed on to the robot
        Args: u_list = list of control commands
        """
        local_cost = 1000.
        itr = 0
        self._rho = self._rho_bkup.copy()
        self._itr += 1

        while (itr < self._max_itr_conv):
            print "Iteration \t", itr
            print "Old u_list \n", u_list

            u_list, xs_samples, traj_prob, delus_samples = self.fun_to_parrallelize(u_list)

            print "New u_list \n", u_list
            
            itr += 1
            
            print "Old rho \n", self._rho
            print "Weighting \n", np.sum(traj_prob, axis=0).squeeze()

            traj_prob = np.ones(self._N)*0.99

            self._rho = np.multiply(self._rho, traj_prob)
            
            print "New rho \n", self._rho
            print "***********************************************************"

            # if KeyboardInterrupt:
            #     break

            print "Modified u_list \t", u_list
            
        return u_list, xs_samples, traj_prob, delus_samples

