import numpy as np
import tensorflow as tf
from aml_opt.policy_opt.mlp_policy_dist import MLPPolicyDist
from aml_opt.policy_opt.diagonal_guassian import DiagonalGaussian


class GaussianMLPPolicy(object):
    def __init__(
            self,
            env,
            tf_sess,
            hidden_sizes=(32, 32),
            learn_std=True,
            init_std=1.0,
            adaptive_std=False,
            std_share_network=False,
            std_hidden_sizes=(32, 32),
            min_std=1e-6,
            std_hidden_nonlinearity='tanh',
            hidden_nonlinearity='tanh',
            output_nonlinearity=None,
            mean_network=None,
            std_network=None,
            dist_cls=DiagonalGaussian,
    ):
        """
        :param env_spec:
        :param hidden_sizes: list of sizes for the fully-connected hidden layers
        :param learn_std: Is std trainable
        :param init_std: Initial std
        :param adaptive_std:
        :param std_share_network:
        :param std_hidden_sizes: list of sizes for the fully-connected layers for std
        :param min_std: whether to make sure that the std is at least some threshold value, to avoid numerical issues
        :param std_hidden_nonlinearity:
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :param std_network: custom network for the output log std
        :return:
        """

        obs_dim    = env._obs_dim
        action_dim = env._action_dim
        self.min_std = min_std

        # create network
        self._policy_network = MLPPolicyDist(
                input_shape=obs_dim,
                output_dim=action_dim,
                hidden_sizes=hidden_sizes,
                hidden_activation=hidden_nonlinearity,
                mu_activation=output_nonlinearity,
                sig_activation =None,

            )

        self._dist = dist_cls(action_dim)

        self._tf_sess = tf_sess

        def f_dist(x_obs, sess=self._tf_sess):

            pol_out = self._policy_network.get_output(x=x_obs, sess=self._tf_sess)

            mean_var    = pol_out[:,:action_dim]
            log_std_var = pol_out[:, action_dim:]

            # if self.min_std is not None:
            #     log_std_var = tf.maximum(log_std_var, np.log(min_std))
            
            return mean_var.flatten(), log_std_var.flatten()

        self._f_dist  = f_dist
        self._a_dim   = action_dim

        self._debug_log = []
        

    def f_dist(self, obs_var):
        return [self._mean_network.output_layer, self._std_network.output_layer]

    def dist_info_sym(self, obs_var, state_info_vars=None):
        # mean_var, log_std_var = L.get_output([self._l_mean, self._l_log_std], obs_var)
        # mean_var = self._mean_network.get_output(obs_var)
        # log_std_var = self._std_network.get_output(obs_var)
        mean_var    = self._policy_network.output_layer[:self._a_dim]
        log_std_var = self._policy_network.output_layer[self._a_dim:]

        if self.min_std is not None:
            log_std_var = tf.maximum(log_std_var, np.log(self.min_std))
        return dict(mean=mean_var, log_std=log_std_var)

    def get_action(self, observation):
        flat_obs = observation.flatten()

        mean, log_std = self._f_dist([flat_obs])

        rnd = np.random.normal(size=mean[0].shape)

        action = rnd * np.exp(log_std) + mean

        return action, dict(mean=mean, log_std=log_std)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        means, log_stds = self._f_dist(flat_obs)
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds)

    def get_reparam_action_sym(self, obs_var, action_var, old_dist_info_vars):
        """
        Given observations, old actions, and distribution of old actions, return a symbolically reparameterized
        representation of the actions in terms of the policy parameters
        :param obs_var:
        :param action_var:
        :param old_dist_info_vars:
        :return:
        """
        new_dist_info_vars = self.dist_info_sym(obs_var, action_var)
        new_mean_var, new_log_std_var = new_dist_info_vars["mean"], new_dist_info_vars["log_std"]
        old_mean_var, old_log_std_var = old_dist_info_vars["mean"], old_dist_info_vars["log_std"]
        epsilon_var = (action_var - old_mean_var) / (tf.exp(old_log_std_var) + 1e-8)
        new_action_var = new_mean_var + epsilon_var * tf.exp(new_log_std_var)
        return new_action_var

    def log_diagnostics(self, paths):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        return self._dist

    def likelihood_sym(self, x_var, dist_info_vars):
        return tf.exp(self.log_likelihood_sym(x_var, dist_info_vars))

    @property
    def output_layer(self):
        return self._policy_network.output_layer


    @property
    def state_info_keys(self):
        """
        Return keys for the information related to the policy's state when taking an action.
        :return:
        """
        return list()

    def get_params(self, **tags):  # adds the list to the _cached_params dict under the tuple key (one)
        """
        Get the list of parameters, filtered by the provided tags.
        Some common tags include 'regularizable' and 'trainable'
        """
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_params:
            self._cached_params[tag_tuple] = self.get_params_internal(**tags)
        return self._cached_params[tag_tuple]

    def get_param_values(self, **tags):
        pass
        

    def set_param_values(self, flattened_params, **tags):
        pass

    def reset(self):
        pass

   


