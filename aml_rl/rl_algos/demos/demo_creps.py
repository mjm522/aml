import numpy as np
from rl_algos.environments.env import Env
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from rl_algos.agents.creps import CREPSOpt
from rl_algos.policy.lin_gauss_policy import LinGaussPolicy

random_state = np.random.RandomState(0)
initial_params = 4.0 * np.ones(1)
n_samples_per_update = 30
variance = 0.03
n_episodes = 200#64
rewards = []
test_contexts = np.arange(-6, 6, 0.1)

env = Env(x0=0., n_samples_per_update=n_samples_per_update, random_state=random_state)

policy = LinGaussPolicy(w_dim=1, context_feature_dim=3, variance=0.03, 
	                    initial_params=initial_params, random_state=random_state)

mycreps = CREPSOpt(entropy_bound=2.0, initial_params=initial_params, num_policy_updates=30, 
                   num_samples_per_update=n_samples_per_update, num_old_datasets=1, 
                   env=env, policy=policy)


for it in range(n_episodes):

    print "Episode \t", it

    mycreps.run()

    policy = mycreps._policy
    
    test_params = np.array([policy.compute_w(np.array([s]), explore=False) for s in test_contexts])
    
    mean_reward = np.mean(
        np.array([env.reward(p, np.array([s]))[0]
                  for p, s in zip(test_params, test_contexts)]))
    
    rewards.append(mean_reward)


plt.plot(rewards)
plt.show()