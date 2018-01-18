import numpy as np
from scipy.signal import lfilter
from utilities import stack_tensor_dict_list, concat_tensor_dict_list, get_feature

def rollout(env, policy, max_path_length=np.inf, speedup=1, always_return_paths=False):
    states = []
    observations = []
    actions = []
    rewards = []
    policy_vars = []
    state = env.reset()
    policy.reset()
    path_length = 0
    features = []

    while path_length < max_path_length:
        
        feature = get_feature(observation=state, reward=0.)
        action, policy_var = policy.get_action(feature)
        state, action, observation, reward = env.step(state, action)
        
        observations.append(observation.flatten())
        states.append(state)
        rewards.append(reward)
        actions.append(action.flatten())
        policy_vars.append(policy_var)

        features.append(get_feature(observation=state, reward=reward))
        
        path_length += 1
        state = observation


    return dict(
        observations= np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        policy_vars=stack_tensor_dict_list(policy_vars),
        features=np.array(features)
    )


class BatchSampler(object):
    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo
        self._num_samples = 20
        self._path_length = 100


    def obtain_samples(self, itr):
        paths = []

        for k in range(self._num_samples):
            paths.append(rollout(self.algo._env, self.algo._policy, self._path_length))
  
        return paths
  

    def process_samples(self, itr, paths):
        baselines = []
        returns = []

        for idx, path in enumerate(paths):
     
            # deltas = path["rewards"] + \
            #          self.algo._discount * path_baselines[1:] - \
            #          path_baselines[:-1]

            deltas = path["rewards"]
            
            #discout cumsum
            path["advantages"] = lfilter([1], [1, float(-self.algo._discount * self.algo._gae_lambda)], deltas[::-1], axis=0)[::-1]
            path["returns"]    = lfilter([1], [1, float(-self.algo._discount)], path["rewards"][::-1], axis=0)[::-1]

            returns.append(path["returns"])

        observations = np.concatenate([path["observations"] for path in paths], axis=0)
        actions = np.concatenate([path["actions"] for path in paths], axis=0)
        rewards = np.concatenate([path["rewards"] for path in paths], axis=0)
        returns = np.concatenate([path["returns"] for path in paths], axis=0)
        advantages = np.concatenate([path["advantages"] for path in paths], axis=0)

        features  = np.concatenate([path['features'] for path in paths], axis=0)

        policy_vars = concat_tensor_dict_list([path["policy_vars"] for path in paths])

        if self.algo._center_adv:
            advantages = (advantages - np.mean(advantages)) / (advantages.std() + 1e-8)

        if self.algo._positive_adv:
            advantages = (advantages - np.min(advantages)) + 1e-8

        average_discounted_return = \
            np.mean([path["returns"][0] for path in paths])

        undiscounted_returns = [sum(path["rewards"]) for path in paths]

        # ent = np.mean(self.algo.policy.distribution.entropy(agent_infos))

        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            returns=returns,
            advantages=advantages,
            policy_vars=policy_vars,
            paths=paths,
            features=features,
        )

        ##the following lines are for the base line linear feature fitting between features and reward
        
        # print("fitting baseline...")
        # if hasattr(self.algo.baseline, 'fit_with_samples'):
        #     self.algo.baseline.fit_with_samples(paths, samples_data)
        # else:
        #     self.algo.baseline.fit(paths)
        # print("fitted")

        return samples_data