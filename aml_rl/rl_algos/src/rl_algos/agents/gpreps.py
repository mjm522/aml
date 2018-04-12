import GPy
import copy
import itertools
import numpy as np
from collections import deque
from aml_io.log_utils import aml_logging
from scipy.optimize import fmin_l_bfgs_b
from rl_algos.utils.utils import logsumexp

"""

Implementation of : C-REPS
Kupcsik, Andras, Marc Peter Deisenroth, Jan Peters, 
Ai Poh Loh, Prahlad Vadakkepat, and Gerhard Neumann. 
"Model-based contextual policy search for data-efficient generalization of robot skills." 
Artificial Intelligence 247 (2017): 415-439.

"""

np.random.seed(123) 

class GPREPSOpt():

    def __init__(self, entropy_bound, num_policy_updates, 
                       num_samples_per_update, num_old_datasets,
                       env, context_model, traj_rollout_model,
                       policy, min_eta=1e-8, num_data_to_collect=20, num_fake_data=30):

        self._logger = aml_logging.get_logger(__name__)

        #epsilon in the algorithm
        self._entropy_bound = entropy_bound
        #pi(w|s) in the algorithm
        self._policy = policy
        #K in the algorithm
        self._num_policy_updates = num_policy_updates
        #number of samples to be used for each update (N)
        self._num_samples_per_update = num_samples_per_update
        #
        self._num_data_to_collect = num_data_to_collect
        #H in the algorithm
        self._num_old_datasets = num_old_datasets
        #to execute the policy
        self._env = env
        #param dim
        self._w_dim = self._policy._w_dim

        #M in the paper
        self._num_fake_data = num_fake_data

        self._min_eta = min_eta

        self._itr = 0

        self._context_model = context_model

        self._traj_model = traj_rollout_model

        #data storing lists  by setting a maximum length we are able to 
        #remember the history as well. 
        self._context_obs_history = deque(maxlen=self._num_samples_per_update)
        self._policy_w_history = deque(maxlen=self._num_samples_per_update)
        self._rewards_history = deque(maxlen=self._num_samples_per_update)


    def sample_context(self):

        return self._context_model.sample()

    def predict_reward(self, s, w):

        new_mu = self._traj_model.predict(s=s, w=w[0])

        return None, new_mu

    def learn_fwd_model(self):

        self._context_model.fit()

        self._traj_model.fit()


    def artificial_data(self):

        #get the context
        s_i = self.sample_context()

        #compute policy params for each context
        w_i = self._policy.compute_w(context=s_i)
        #execute the policy in the environment
        tau_i, R_sw_i = self.predict_reward(s=s_i, w=w_i)

        #store the data
        self._context_obs_history.append(self._policy.transform_context(s_i))
        self._policy_w_history.append(w_i)
        self._rewards_history.append(R_sw_i[0])

        #sample count
        self._itr += 1 


    def opt_dual_function(self):

        S = np.asarray(self._context_obs_history)
        w = np.asarray(self._policy_w_history)
        R = np.asarray(self._rewards_history).flatten()

        # import matplotlib.pyplot as plt

        # data_S = np.asarray(self._context_data)
        # data_R = np.asarray(self._reward_data)

        # plt.figure("S")
        # plt.plot(S[:,0], 'g')
        # plt.plot(data_S[:,0], 'r')

        # plt.figure("R")
        # plt.plot(data_R, 'r')
        # plt.plot(R, 'g')

        # plt.show(False)
        
        n_samples_per_update = len(R)

        # Definition of the dual function
        def g(x):  # Objective function
            eta = x[0]
            theta = x[1:]
            return (eta * self._entropy_bound + theta.T.dot(S.mean(axis=0)) +
                    eta * logsumexp((R - theta.dot(S.T)) / eta,
                                    b=1.0 / n_samples_per_update))

        # Lower bound for Lagrange parameters eta and theta
        bounds = np.vstack(([[self._min_eta, None]], np.tile(None, (S.shape[1], 2))))
        # Start point for optimization
        x0 = [1] + [1] * S.shape[1]

        # Perform the actual optimization of the dual function
        #r = NLP(g, x0, lb=lb).solve('ralg', iprint=-10)
        r = fmin_l_bfgs_b(g, x0, approx_grad=True, bounds=bounds)
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

    def update_policy(self, weights):

        self._policy.fit(weights=weights, 
                         S=np.asarray(self._context_obs_history), 
                         B=np.asarray(self._policy_w_history))


    def collect_data(self):

        if len(self._context_model._data) < self._num_data_to_collect:

            num_trials = 20
        else:
            num_trials = 1

        for k in range(num_trials):

            #get the context
            s_k = self._env.context()
            #compute policy params for each context
            w_k = self._policy.compute_w(s_k)
            #execute the policy in the environment
            tau_k, R_sw_k = self._env.execute_policy(w_k, s_k)

            self._context_model.add_data(datum=s_k)

            self._traj_model.add_data(w=w_k, r=R_sw_k)

            if num_trials == 20:

                self._context_obs_history.append(self._policy.transform_context(s_k))
                self._policy_w_history.append(w_k)
                self._rewards_history.append(R_sw_k)
    
        if num_trials == 20:

            weights, eta, theta = self.opt_dual_function()

            self.update_policy(weights)


    def run(self):
        #main loop

        self.collect_data()

        self.learn_fwd_model()

        for k in range(self._num_samples_per_update):

            self.artificial_data()

            if (self._itr%self._num_policy_updates == 0):

                weights, eta, theta = self.opt_dual_function()

                self.update_policy(weights)

        return self._policy

            
