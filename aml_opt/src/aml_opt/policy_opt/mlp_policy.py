import numpy as np
import tensorflow as tf
from aml_opt.policy_opt.mlp_policy_dist import MLPPolicyDist

class MLPPolicy(object):
    def __init__(self, tf_sess, config):

        self._min_std = config['min_std']

        # create network
        self._policy_network = MLPPolicyDist(
                input_shape      =config['feature_dim'],
                output_dim       =config['output_dim'],
                hidden_sizes     =config['hidden_sizes'],
                hidden_activation= config['hidden_nonlinearity'],
                mu_activation    = config['output_nonlinearity'],
                sig_activation   =None,

            )


        self._tf_sess = tf_sess
        self._action_dim   = config['output_dim']/2

        def f_dist(x_obs, sess=self._tf_sess):

            pol_out = self._policy_network.get_output(x=x_obs, sess=self._tf_sess)

            mean_var    = pol_out[:,:self._action_dim]
            log_std_var = pol_out[:, self._action_dim:]

            # if self._min_std is not None:
            #     log_std_var = tf.maximum(log_std_var, np.log(self._min_std))
            
            return mean_var.flatten(), log_std_var.flatten()

        self._f_dist  = f_dist
        

    def f_dist(self, obs_var):
        return [self._mean_network.output_layer, self._std_network.output_layer]

    def dist_vars_sym(self, obs_var):

        mean_var    = self._policy_network.output_layer[:,:self._action_dim]
        log_std_var = self._policy_network.output_layer[:,self._action_dim:]

        if self._min_std is not None:
            log_std_var = tf.maximum(log_std_var, np.log(self._min_std))
            
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

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        old_means    = old_dist_info_vars["mean"]
        old_log_stds = old_dist_info_vars["log_std"]
        new_means    = new_dist_info_vars["mean"]
        new_log_stds = new_dist_info_vars["log_std"]
        """
        Compute the KL divergence of two multivariate Gaussian distribution with
        diagonal covariance matrices
        """
        old_std = tf.exp(old_log_stds)
        new_std = tf.exp(new_log_stds)
        # means: (N*A)
        # std: (N*A)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
        # ln(\sigma_2/\sigma_1)
        numerator = tf.square(old_means - new_means) + \
                    tf.square(old_std) - tf.square(new_std)
        denominator = 2 * tf.square(new_std) + 1e-8
        return tf.reduce_sum(
            numerator / denominator + new_log_stds - old_log_stds, axis=-1)

    def kl(self, old_dist_info, new_dist_info):
        old_means    = old_dist_info["mean"]
        old_log_stds = old_dist_info["log_std"]
        new_means    = new_dist_info["mean"]
        new_log_stds = new_dist_info["log_std"]
        """
        Compute the KL divergence of two multivariate Gaussian distribution with
        diagonal covariance matrices
        """
        old_std = np.exp(old_log_stds)
        new_std = np.exp(new_log_stds)
        # means: (N*A)
        # std: (N*A)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
        # ln(\sigma_2/\sigma_1)
        numerator = np.square(old_means - new_means) + \
                    np.square(old_std) - np.square(new_std)
        denominator = 2 * np.square(new_std) + 1e-8
        return np.sum(
            numerator / denominator + new_log_stds - old_log_stds, axis=-1)

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        logli_new = self.log_likelihood_sym(x_var, new_dist_info_vars)
        logli_old = self.log_likelihood_sym(x_var, old_dist_info_vars)
        return tf.exp(logli_new - logli_old)

    def log_likelihood_sym(self, x_var, dist_vars):
        means = dist_vars["mean"]
        log_stds = dist_vars["log_std"]

        zs = (x_var - means) / tf.exp(log_stds)
        return - tf.reduce_sum(log_stds, axis=-1) - \
               0.5 * tf.reduce_sum(tf.square(zs), axis=-1) - \
               0.5 * means.shape[-1].value * np.log(2 * np.pi)

    def sample(self, dist_info):
        means = dist_info["mean"]
        log_stds = dist_info["log_std"]
        rnd = np.random.normal(size=means.shape)
        return rnd * np.exp(log_stds) + means

    def log_likelihood(self, xs, dist_info):
        means = dist_info["mean"]
        log_stds = dist_info["log_std"]
        zs = (xs - means) / np.exp(log_stds)
        return - np.sum(log_stds, axis=-1) - \
               0.5 * np.sum(np.square(zs), axis=-1) - \
               0.5 * means.shape[-1] * np.log(2 * np.pi)

    def entropy(self, dist_info):
        log_stds = dist_info["log_std"]
        return np.sum(log_stds + np.log(np.sqrt(2 * np.pi * np.e)), axis=-1)

    def entropy_sym(self, dist_info_var):
        log_std_var = dist_info_var["log_std"]
        return tf.reduce_sum(log_std_var + tf.log(np.sqrt(2 * np.pi * np.e)), axis=-1)

    @property
    def dist_keys(self):
        return ["mean", "log_std"]

    def reset(self):
        pass

   


