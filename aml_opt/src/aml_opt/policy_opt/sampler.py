import numpy as np
from scipy.signal import lfilter

def rollout(env, agent, max_path_length=np.inf, speedup=1, always_return_paths=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o.flatten())
        rewards.append(r)
        actions.append(a.flatten())
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o

    return dict(
        observations= np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        agent_infos=None,#stack_tensor_dict_list(agent_infos),
        env_infos=None,#stack_tensor_dict_list(env_infos),
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
            paths.append(rollout(self.algo.env, self.algo.policy, self._path_length))
  
        return paths
  

    def process_samples(self, itr, paths):
        baselines = []
        returns = []

        if hasattr(self.algo.baseline, "predict_n"):
            all_path_baselines = self.algo.baseline.predict_n(paths)
        else:
            all_path_baselines = [self.algo.baseline.predict(path) for path in paths]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            #discout cumsum
            path["advantages"] = lfilter([1], [1, float(-self.algo.discount * self.algo.gae_lambda)], deltas[::-1], axis=0)[::-1]
            path["returns"]    = lfilter([1], [1, float(-self.algo.discount)], path["rewards"][::-1], axis=0)[::-1]

            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        # ev = explained_variance_1d( np.concatenate(baselines),np.concatenate(returns) )

        observations = np.concatenate([path["observations"] for path in paths], axis=0)
        actions = np.concatenate([path["actions"] for path in paths], axis=0)
        rewards = np.concatenate([path["rewards"] for path in paths], axis=0)
        returns = np.concatenate([path["returns"] for path in paths], axis=0)
        advantages = np.concatenate([path["advantages"] for path in paths], axis=0)

        # env_infos = concat_tensor_dict_list([path["env_infos"] for path in paths])
        # agent_infos = concat_tensor_dict_list([path["agent_infos"] for path in paths])

        if self.algo.center_adv:
            advantages = (advantages - np.mean(advantages)) / (advantages.std() + 1e-8)

        if self.algo.positive_adv:
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
            env_infos=None,#env_infos,
            agent_infos=None,#agent_infos,
            paths=paths,
        )
        
        print("fitting baseline...")
        if hasattr(self.algo.baseline, 'fit_with_samples'):
            self.algo.baseline.fit_with_samples(paths, samples_data)
        else:
            self.algo.baseline.fit(paths)
        print("fitted")

        return samples_data