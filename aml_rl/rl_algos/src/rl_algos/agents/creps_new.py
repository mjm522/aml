import copy
import itertools
import numpy as np
from collections import deque
from rl_algos.utils.utils import lpf
from scipy.optimize import fmin_l_bfgs_b
from rl_algos.utils.utils import logsumexp
from rl_algos.policy.lin_gauss_policy import LinGaussPolicy
from aml_playground.var_imp_reps.utils.plot_util import exp_plot

"""

Implementation of : C-REPS
Kupcsik, Andras, Marc Peter Deisenroth, Jan Peters, 
Ai Poh Loh, Prahlad Vadakkepat, and Gerhard Neumann. 
"Model-based contextual policy search for data-efficient generalization of robot skills." 
Artificial Intelligence 247 (2017): 415-439.

"""

np.random.seed(123) 

class CREPSOpt():

    def __init__(self, entropy_bound, initial_params, num_policy_updates, 
                       num_samples_per_update, num_old_datasets, 
                       env, policy, time_steps=100, min_eta=1e-8, transform_context=True,
                       policy_per_time_step=True):
        
        self._policy_per_time_step = policy_per_time_step
        #epsilon in the algorithm
        self._entropy_boud = entropy_bound

        self._time_steps = time_steps

        #initial params
        self._w_init = initial_params
        #pi(w|s) in the algorithm
        self._policy = policy
        #K in the algorithm
        self._num_policy_updates = num_policy_updates
        #number of samples to be used for each update (N)
        self._num_samples_per_update = num_samples_per_update
        #H in the algorithm
        self._num_old_datasets = num_old_datasets
        #to execute the policy
        self._env = env
        #param dim
        if self._policy_per_time_step:
            self._w_dim = self._policy[0]._w_dim
        else:
            self._w_dim = self._policy._w_dim
        self._c_dim = len(self._env.context())

        self._min_eta = min_eta

        self._itr = 0

        self._transform_context = transform_context

        #data storing lists  by setting a maximum length we are able to 
        #remember the history as well. 
        if self._policy_per_time_step:
            self._context_obs_history = np.empty([self._c_dim, self._time_steps, self._num_samples_per_update]) #[deque(maxlen=self._time_steps) for _ in range(self._num_samples_per_update)]
            self._policy_w_history = np.empty([self._w_dim, self._time_steps, self._num_samples_per_update])#[deque(maxlen=self._time_steps) for _ in range(self._num_samples_per_update)]
            self._rewards_history = np.empty([1, self._time_steps, self._num_samples_per_update])#[deque(maxlen=self._time_steps) for _ in range(self._num_samples_per_update)]
        else:
            self._context_obs_history = np.empty([self._c_dim, self._num_samples_per_update*self._time_steps]) #[deque(maxlen=self._time_steps) for _ in range(self._num_samples_per_update)]
            self._policy_w_history = np.empty([self._w_dim, self._num_samples_per_update*self._time_steps])#[deque(maxlen=self._time_steps) for _ in range(self._num_samples_per_update)]
            self._rewards_history = np.empty([1, self._num_samples_per_update*self._time_steps])#[deque(maxlen=self._time_steps) for _ in range(self._num_samples_per_update)]


    def add_data(self, sample_no, jnt_space = False):

        self._env._reset()

        #execute the policy in the environment
        tau_i, R_sw_i = self._env.execute_policy(self._policy, jnt_space = jnt_space, policy_per_time_step = self._policy_per_time_step)

        
        if (self._itr%self._num_policy_updates == 0):
            cla = True
        else:
            cla = False
        
        # exp_plot(rewards=None, reward_traj=R_sw_i, traj_draw=tau_i, figsize=(5,5), cla=cla)

        for i, (s_i, w_i, R_i) in enumerate(zip(tau_i['contexts'], tau_i['params'], R_sw_i['reward_traj'])):
            #store the data
            if self._policy_per_time_step:
                if self._transform_context:
                    self._context_obs_history[:, i, sample_no] = self._policy.transform_context(s_i)
                else:
                    self._context_obs_history[:, i, sample_no] = s_i
                
                self._policy_w_history[:, i, sample_no] = w_i

                self._rewards_history[:, i, sample_no] = R_i

            else:
                if self._transform_context:
                    self._context_obs_history[:, (self._time_steps*sample_no)+i] = self._policy.transform_context(s_i)
                else:
                    self._context_obs_history[:, (self._time_steps*sample_no)+i] = s_i
                
                self._policy_w_history[:, (self._time_steps*sample_no)+i] = w_i

                self._rewards_history[:, (self._time_steps*sample_no)+i] = R_i

        #sample count
        self._itr += 1


    def opt_dual_function(self, time_step):

        if time_step is not None:
            S = self._context_obs_history[:, time_step, :].T
            w = self._policy_w_history[:, time_step, :].T
            R = self._rewards_history[:, time_step, :].flatten()
        else:
            S = self._context_obs_history[:, :].T
            w = self._policy_w_history[:, :].T
            R = self._rewards_history[:, :].flatten()

        n_samples_per_update = len(R)

        # R = (R - R.min()) / (R.max() - R.min())

        # Definition of the dual function
        def g(x):  # Objective function
            eta = x[0]
            theta = x[1:]

            s_mean = S.mean(axis=0)
            R_over_eta = (R-theta.dot(S.T))/eta
            log_sum_exp = logsumexp(R_over_eta, b=1.0/n_samples_per_update)
            Z = np.exp(R_over_eta - R_over_eta.max())
            Z_sum = Z.sum()

            f = eta*(self._entropy_boud + log_sum_exp) + theta.T.dot(s_mean)
            d_eta = self._entropy_boud + log_sum_exp - Z.dot(R_over_eta)/Z_sum
            d_theta = s_mean - Z.dot(S)/Z_sum

            return f, np.append(d_eta, d_theta)

            # return (eta * self._entropy_boud + theta.T.dot(S.mean(axis=0)) +
            #         eta * logsumexp((R - theta.dot(S.T)) / eta,
            #                         b=1.0 / n_samples_per_update))

        # Lower bound for Lagrange parameters eta and theta
        bounds = np.vstack(([[self._min_eta, None]], np.tile(None, (S.shape[1], 2))))
        # Start point for optimization
        x0 = [1] + [1] * S.shape[1]

        # Perform the actual optimization of the dual function
        #r = NLP(g, x0, lb=lb).solve('ralg', iprint=-10)
        # r = fmin_l_bfgs_b(g, x0, approx_grad=True, bounds=bounds)
        r = fmin_l_bfgs_b(g, x0, bounds=bounds)
        # Fetch optimal lagrangian parameter eta. Corresponds to a temperature
        # of a softmax distribution
        eta = r[0][0]
        # Fetch optimal vale of vector theta which determines the context
        # dependent baseline
        theta = r[0][1:]

        # Determine weights of individual samples based on the their return,
        # the optimal baseline theta.dot(\phi(s)) and the "temperature" eta
        log_d = (R - theta.dot(S.T)) / eta
        # Numerically stable softmax version of the weights. Note that
        # this does neither changes the solution of the weighted least
        # squares nor the estimation of the covariance.
        weights = np.exp(log_d - log_d.max())
        weights /= weights.sum()

        return weights, eta, theta

    def update_policy(self, weights, time_step = None):

        if time_step is not None:
            self._policy[time_step].fit(weights=weights, 
                                        S=self._context_obs_history[:, time_step, :].T, 
                                        B=self._policy_w_history[:, time_step, :].T)
        else:
            self._policy.fit(weights=weights, 
                                        S=self._context_obs_history[:, :].T, 
                                        B=self._policy_w_history[:, :].T)


    def smooth_policy(self):

        if self._policy_per_time_step:
            len_policy = len(self._policy)
            w_r, w_c = self._policy[0]._w.shape
            s_r, s_c = self._policy[0]._sigma.shape
            w_traj = np.zeros([w_r, w_c, len_policy])
            sigma_traj = np.zeros([s_r, s_c, len_policy])

            for k, pol in enumerate(self._policy):
                w_traj[:,:,k] = pol._w
                sigma_traj[:,:,k]=pol._sigma

            for r in range(w_r):
                for c in range(w_c):
                    w_traj[r,c,:] = lpf(w_traj[r,c,:])

            for r in range(s_r):
                for c in range(s_c):
                    sigma_traj[r,c,:] = lpf(sigma_traj[r,c,:])

            for k in range(len_policy):
                self._policy[k]._w = w_traj[:,:,k]
                self._policy[k]._sigma = sigma_traj[:,:,k]
        else:
            w_r, w_c = self._policy._w.shape
            s_r, s_c = self._policy._sigma.shape
            w_traj = np.zeros([w_r, w_c])
            sigma_traj = np.zeros([s_r, s_c])

            for r in range(w_r):
                for c in range(w_c):
                    w_traj[r,c] = lpf(w_traj[r,c])

            for r in range(s_r):
                for c in range(s_c):
                    sigma_traj[r,c] = lpf(sigma_traj[r,c])

            self._policy._w = w_traj
            self._policy._sigma = sigma_traj



    def run(self, smooth_policy=False, jnt_space = False):
        #main loop
        for k in range(self._num_samples_per_update): #30

            self.add_data(sample_no=k, jnt_space = jnt_space)

            if (self._itr%self._num_policy_updates == 0): #15

                if self._policy_per_time_step:

                    for i in range(self._time_steps):

                        weights, eta, theta = self.opt_dual_function(time_step=i)

                        self.update_policy(weights=weights, time_step=i)
                else:
                    weights, eta, theta = self.opt_dual_function(time_step=None)

                    self.update_policy(weights=weights, time_step=None)

            if smooth_policy:
                self.smooth_policy()

        return self._policy

            
