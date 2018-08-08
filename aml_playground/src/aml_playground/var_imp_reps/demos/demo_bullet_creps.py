import numpy as np
import matplotlib.pyplot as plt
from rl_algos.agents.creps_new import CREPSOpt
from aml_io.io_tools import save_data, load_data
from rl_algos.policy.lin_gauss_policy import LinGaussPolicy
from aml_rl_envs.sawyer.sawyer_var_imp_env import SawyerEnv
# from aml_playground.peg_in_hole_reps.utilities.plot_data import plot_data
from aml_playground.var_imp_reps.exp_params.experiment_var_imp_params import exp_params

random_state = np.random.RandomState(0)
initial_params = .001 * np.ones(exp_params['gpreps_params']['w_dim'])
num_policy_updates = 20
n_samples_per_update = 20
variance = 0.03
n_episodes = 160
time_steps=100

rewards = []

env_params = exp_params['env_params']
env_params['renders'] = False

env = SawyerEnv(env_params)

policy = [ LinGaussPolicy(w_dim=exp_params['gpreps_params']['w_dim'], context_feature_dim=9, variance=0.03, 
                         initial_params=initial_params, bounds=exp_params['gpreps_params']['w_bounds'], random_state=random_state, transform=False) for _ in range(time_steps)]

mycreps = CREPSOpt(entropy_bound=2.0, initial_params=initial_params, num_policy_updates=num_policy_updates, 
                   num_samples_per_update=n_samples_per_update, num_old_datasets=1, 
                   env=env, policy=policy, transform_context=False)

plt.ion()

data = []

it = 0
while it < (n_episodes):

    try:

        print "Episode \t", it

        policy = mycreps.run(smooth_policy=True, jnt_space = True)

        env._reset()

        traj_draw, reward = env.execute_policy(policy=policy, explore=False, jnt_space = True)
        
        rewards.append(reward['total'])

        traj_draw['mean_reward'] = reward['total']

        data.append(traj_draw)

        it += 1

        plt.plot(rewards, 'b')
        plt.draw()
        plt.pause(0.0001)

    except KeyboardInterrupt:
        break

c = raw_input("Save data? (y/N)")

if c == 'y':
    print "Saving to %s"%exp_params['param_file_name']
    save_data(data, exp_params['param_file_name'])

raw_input("press any key to continue")