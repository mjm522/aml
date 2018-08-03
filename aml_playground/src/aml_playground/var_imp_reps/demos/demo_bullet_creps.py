import numpy as np
import matplotlib.pyplot as plt
from rl_algos.agents.creps_new import CREPSOpt
from aml_io.io_tools import save_data, load_data
from rl_algos.policy.lin_gauss_policy import LinGaussPolicy
from aml_rl_envs.sawyer.sawyer_var_imp_env import SawyerEnv
from aml_playground.var_imp_reps.exp_params.experiment_var_imp_params import exp_params

random_state = np.random.RandomState(0)
initial_params = 4.0 * np.ones(6)
n_samples_per_update = 100
variance = 0.03
n_episodes = 200
rewards = []

env_params = exp_params['env_params']
env_params['renders'] = False

env = SawyerEnv(env_params)

policy = LinGaussPolicy(w_dim=6, context_feature_dim=9, variance=0.03, 
                        initial_params=initial_params, random_state=random_state, transform=False)

mycreps = CREPSOpt(entropy_bound=2.0, initial_params=initial_params, num_policy_updates=30, 
                   num_samples_per_update=n_samples_per_update, num_old_datasets=1, 
                   env=env, policy=policy, transform_context=False)

plt.ion()

for it in range(n_episodes):

    print "Episode \t", it

    mycreps.run()

    policy = mycreps._policy

    mean_reward = env._penalty['total']
    
    rewards.append(mean_reward)

    plt.plot(rewards, 'b')
    plt.draw()
    plt.pause(0.0001)

raw_input("press any key to continue")