import os
import GPy
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from rl_algos.agents.creps import CREPSOpt
from rl_algos.agents.gpreps import GPREPSOpt
from aml_io.io_tools import save_data, load_data
from aml_opt.pi_cmd_opt.pi_cmd_opt import PICmdOpt
from aml_opt.pi_traj_opt.pi_traj_opt import PITrajOpt
from rl_algos.policy.lin_gauss_policy import LinGaussPolicy
from rl_algos.forward_models.context_model import ContextModel
from rl_algos.forward_models.traj_rollout_model import TrajRolloutModel
from aml_playground.sawyer_spring.envs.ee_env import EEEnv

# np.random.seed(42)
random.seed(1337)
np.random.seed(1337)

interval = np.sqrt(3)

ee_x = np.linspace(0,2,40)

k=2.5
spring_x = np.hstack([np.ones(10)*0, np.linspace(0,2,20), np.ones(10)*0])
spring_force = spring_x*k
spring_force = savgol_filter(x=spring_force, window_length=9, polyorder=2)

data = {}
data['X'] = np.hstack([ ee_x[0:len(ee_x)-1][:,None], spring_force[0:len(spring_force)-1][:,None] ])
data['Y'] = spring_force[1:len(spring_force)][:,None]


class Env():

    def __init__(self):
        # create simple GP Model
        self._model = GPy.models.GPRegression(data['X'], data['Y'])
        # set the lengthscale to be something sensible (defaults to 1)
        self._model.kern.lengthscale = 10.
        self._model.optimize('bfgs', max_iters=200)

    def context(self):

        x_test = random.uniform(0,2)
        X_test = np.array([x_test, x_test*k])[None, :]

        mean, var = self._model.predict(X_test)

        return mean

    def reward(self, w, s):
        return 1./((s-w*(s/k))**2+0.0001)

    def execute_policy(self, w, s, **kwargs):
        
        return None, self.reward(w, s)


    def _reset(self):

        pass


##############################GPREPS#########################################################3
# rewards = []
# random_state = np.random.RandomState(0)
# initial_params = 1.0 * np.ones(2)
# n_samples_per_update = 30
# variance = 0.03
# n_episodes = 64
# w_dim = len(initial_params)
# num_samples_fwd_data = 50
# test_contexts = np.arange(0, 6, 0.1)

# env = EEEnv(goal=5, random_state=random_state)

# policy = LinGaussPolicy(w_dim=w_dim, context_feature_dim=3, variance=0.03, initial_params=initial_params, random_state=random_state)

# context_model = ContextModel(context_dim=1, 
#                             num_data_points=num_samples_fwd_data)

# traj_model = TrajRolloutModel(w_dim=w_dim, x_dim=2, cost=env.reward, 
#                             context_model=context_model, num_data_points=num_samples_fwd_data)

# mycreps = GPREPSOpt(entropy_bound=2.0, num_policy_updates=25, 
#                     num_samples_per_update=n_samples_per_update, num_old_datasets=1, env=env, 
#                     context_model=context_model, traj_rollout_model=traj_model,
#                     policy=policy)


# for it in range(n_episodes):

#     print "Episode \t", it

#     mycreps.run()

#     policy = mycreps._policy
    
#     test_params = np.array([policy.compute_w(np.array([s]), explore=False) for s in test_contexts])
    
#     mean_reward = np.mean(
#         np.array([env.reward(p, np.array([s]))[0]
#                   for p, s in zip(test_params, test_contexts)]))
    
#     rewards.append(mean_reward)

# plt.figure('params')
# plt.plot(test_params[:,0], 'g')
# plt.plot(test_contexts, 'r')
# plt.figure('rewards')
# plt.plot(rewards)
# plt.show()
###################################################################################################33

env = Env()


# ee_x_test = np.linspace(0,2,40)
# spring_x_test = np.hstack([np.ones(10)*0, np.linspace(0,2,20), np.ones(10)*2])
# X_test = np.hstack([ ee_x_test[:,None], spring_x_test[:,None]*k ])

# mean, var = env._model.predict(X_test)
# lower_bound = mean[:,0] - interval*var[:,0]
# upper_bound = mean[:,0] + interval*var[:,0]
# # plot the shaded range of the confidence intervals

# plt.figure('true')
# plt.fill_between(range(mean.shape[0]), lower_bound, upper_bound, color=[0.5,0.5,0.5], alpha=.5)
# plt.plot(data['Y'], 'r', label='true')
# plt.plot(mean, 'g', label='predicted')

# trial = np.zeros(100)
# for i in range(100):
#     trial[i] = env.context()

# plt.figure('trial')
# plt.plot(trial)
# plt.show()


# random_state = np.random.RandomState(0)
# initial_params = 1.0 * np.ones(1)
# n_samples_per_update = 30
# variance = 0.03
# n_episodes = 200#64
# rewards = []
# test_contexts = np.arange(0, 2, 0.1)*k

# env = Env()

# policy = LinGaussPolicy(w_dim=1, context_feature_dim=3, variance=0.03, 
#                         initial_params=initial_params, random_state=random_state)

# mycreps = CREPSOpt(entropy_bound=2.0, initial_params=initial_params, num_policy_updates=30, 
#                    num_samples_per_update=n_samples_per_update, num_old_datasets=1, 
#                    env=env, policy=policy)


# for it in range(n_episodes):

#     print "Episode \t", it

#     mycreps.run()

#     test_params = np.array([mycreps._policy.compute_w(np.array([s]), explore=False) for s in test_contexts])
    
#     mean_reward = np.mean(
#         np.array([env.reward(p, np.array([s]))[0]
#                   for p, s in zip(test_params, test_contexts)]))
    
#     rewards.append(mean_reward)

# plt.figure('params')
# plt.plot(test_params, 'g')
# plt.plot(test_contexts, 'r')

# plt.figure('rewards')
# plt.plot(rewards)
# plt.show()

# ctrl_constraints={
# 'min':[0.],
# 'max':[4.],
# }

# config = {
#     'max_iterations':200,
#     'dt': 0.01,
#     'N':15,
#     'h':10.,
#     'rho': 1.5,#10.7,#0.2
#     'K':10,
#     'cmd_dim': 1,
#     'state_dim': 2,
#     'random_seed': 123,
#     'init_policy': 'use_random',
#     'ctrl_constraints':ctrl_constraints,
#     }

# class OptimCmd(PICmdOpt):

#     def __init__(self, config):

#         PICmdOpt.__init__(self, config)
#         self._model = env._model

#     def dynamics(self, x, u, dt=0.01):
        
#         x_test = np.hstack([x, x*k])[None,:]

#         mean, var = self._model.predict(x_test)

#         return mean

#     def cost(self, x, u, du, t):
#         s=k*x
#         return -10.* 1./((s-u*(s/k))**2+0.0001)


#     def forward_rollout(self, u_list, k):

#         xs_samples = np.zeros((self._N, self._state_dim)) # Forward trajectory samples
#         delus_samples = np.zeros((self._N, self._cmd_dim))
        
#         ss = np.zeros(self._N) # Costs

#         xs_samples[0,:] = self._x0.copy()

#         for t in range(0, self._N):
            
#             x_prev = xs_samples[t,:]
            
#             du = self.calc_delu(t)

#             x_next  = self.dynamics(x_prev, u_list[t,:] + du)

#             xs_samples[t,:] = x_next

#             ss[t] += ss[t] + self.cost(x_next, u_list[t,:], du, t+1)
            
#             if ss[t] < 1e-4 or np.isnan(ss[t]):
#                 ss[t] = 0.
            
#             delus_samples[t,:] = du

#         return xs_samples, ss, delus_samples


# def main():

#     pi_cmd = OptimCmd(config=config)

#     u_list = np.random.randn(config['N'], config['cmd_dim'])

#     u_list, xs_samples, traj_prob, delus_samples = pi_cmd.run(u_list)

#     # x_final = np.zeros([config['N']+1, config['state_dim']])

#     # x_final[0, :] = pi_cmd._x0.copy()

#     # for k in range(config['N']):
#     #     x_final[k+1, :] =  pi_cmd.dynamics(x_final[k, :], u_list[:,k])

#     # plt.plot(x_final[:,0], x_final[:,1])
#     plt.plot(u_list[:,0])
#     plt.show()


# if __name__ == '__main__':
#     main()


sg ={
'start':np.array([0, 0]),
'goal':np.array([0, 0]),
}

state_constraints={
'min':[0., 0.],
'max':[4., 4.],
}


config = {
    'timesteps':40,
    'no_rollouts':100,
    'h':10,
    'gain':1e-1,
    'max_iter':1000,
    'state_constraints':state_constraints,
    'start_goal':sg,
    'init_traj':None,
    'cost_fn':None,
    'smooth_traj':True,
}


class TrajCost():
    def __init__(self):
        non_linear_spring = 2.*np.sin(spring_x)
        self._goal_traj = spring_force - non_linear_spring
        plt.plot(self._goal_traj)
        plt.show()
        self._x_traj = spring_x
        self._dx_traj =  np.hstack([0, np.diff(self._x_traj)])

    def __call__(self, traj):
        return self.get(traj)

    def get(self, traj):

        Kptraj = traj[:,0]
        Kdtraj = traj[:,1]

        cost = (self._goal_traj - np.multiply(self._x_traj, Kptraj) - np.multiply(self._dx_traj, Kdtraj) )**2
        cost += 0.15*Kptraj**2
        cost += 0.15*Kdtraj**2

        # cost = cost.dot(cost)

        cost = cost - np.min(cost)

        cost = 1./ (1. + np.exp(-cost))

        cost = cost - np.min(cost)

        return cost


class LowCostPathFinder(object):

    def __init__(self, config):

        config['start'] = config['start_goal']['start']
        config['goal']  = config['start_goal']['goal']
        self._x_min, self._y_min = config['state_constraints']['min']
        self._x_max, self._y_max = config['state_constraints']['max']

        self._x_traj = spring_x
        self._dx_traj =  np.hstack([0, np.diff(self._x_traj)])

        self._cost_fn = TrajCost()

        self._opt = PITrajOpt(config, self._cost_fn, self.visualize)
        self._init_traj = self._opt._init_traj

        # plt.plot(self._init_traj[0,:], self._init_traj[1,:], color='g', linewidth=3)
        # plt.plot(self._init_traj[0,:], self._init_traj[1,:], color='g', linewidth=3)
        plt.ion()
  
    def visualize(self, traj, traj_samples=None):
        Kptraj = traj[:,0]
        Kdtraj = traj[:,1]

        result_traj = np.multiply(self._x_traj, Kptraj) + np.multiply(self._dx_traj, Kdtraj)
        goal_traj = self._cost_fn._goal_traj
        
        # if (traj_samples is not None):
        #     for i in range(traj_samples.shape[0]):
        #         plt.plot(self._opt.savitsky_gollay_filter(traj_samples[i,:,0].flatten()), 
        #                  self._opt.savitsky_gollay_filter(traj_samples[i,:,1].flatten()), alpha=0.2)

        # plt.plot(Kptraj.flatten(), color='g', linewidth=2.)
        # plt.plot(Kdtraj.flatten(), color='r', linewidth=2.)
        plt.figure('force')
        plt.clf()
        plt.plot(result_traj, color='g', linewidth=2.)
        plt.plot(goal_traj, color='r', linewidth=2.)

        plt.figure('params')
        plt.clf()
        plt.plot(Kptraj, color='g', linewidth=2.)
        plt.plot(Kdtraj, color='r', linewidth=2.)

        # plt.xlim(self._x_min, self._x_max)
        # plt.ylim(self._y_min, self._y_max)
        plt.draw()
        plt.pause(0.00001)

    def run(self):
        self._opt.run()


def main():

    lcpf = LowCostPathFinder(config=config)
    lcpf.run()

    raw_input("press enter to exit")

if __name__ == '__main__':
    main()