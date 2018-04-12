import numpy as np
from rl_algos.environments.env import Env
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from rl_algos.agents.gpreps import GPREPSOpt
from rl_algos.policy.lin_gauss_policy import LinGaussPolicy
from rl_algos.forward_models.context_model import ContextModel
from rl_algos.forward_models.traj_rollout_model import TrajRolloutModel

rewards = []
random_state = np.random.RandomState(0)
initial_params = 4.0 * np.ones(1)
n_samples_per_update = 30
variance = 0.03
n_episodes = 64
w_dim = len(initial_params)
num_samples_fwd_data = 50
test_contexts = np.arange(-6, 6, 0.1)


env = Env(x0=0., n_samples_per_update=n_samples_per_update, random_state=random_state)

policy = LinGaussPolicy(w_dim=w_dim, context_feature_dim=3, variance=0.03, initial_params=initial_params, random_state=random_state)

context_model = ContextModel(context_dim=1, 
                            num_data_points=num_samples_fwd_data)

traj_model = TrajRolloutModel(w_dim=w_dim, x_dim=2, cost=env.reward, 
                            context_model=context_model, num_data_points=num_samples_fwd_data)

mycreps = GPREPSOpt(entropy_bound=2.0, num_policy_updates=25, 
                    num_samples_per_update=n_samples_per_update, num_old_datasets=1, env=env, 
                    context_model=context_model, traj_rollout_model=traj_model,
                    policy=policy)


for it in range(n_episodes):

    mycreps.run()

    policy = mycreps._policy
    
    test_params = np.array([policy.compute_w(np.array([s]), explore=False) for s in test_contexts])
    
    mean_reward = np.mean(
        np.array([env.reward(p, np.array([s]))[0]
                  for p, s in zip(test_params, test_contexts)]))
    
    rewards.append(mean_reward)


plt.plot(rewards)
plt.show()