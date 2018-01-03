import numpy as np
import scipy.optimize
import tensorflow as tf
from sampler import BatchSampler
from utilities import TfFunction

class REPS(object):

    def __init__(self, env, policy, tf_sess, baseline, discount=0.9, gae_lambda=0.9, epsilon=0.5, L2_reg_dual=0., L2_reg_loss=0., max_opt_itr=50):
        """
        :param epsilon: Max KL divergence between new policy and old policy.
        :param L2_reg_dual: Dual regularization
        :param L2_reg_loss: Loss regularization
        :param max_opt_itr: Maximum number of batch optimization iterations.
        :param optimizer: Module path to the optimizer. It must support the same interface as
        scipy.optimize.fmin_l_bfgs_b.
        :return:
        """
        self.epsilon = epsilon
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.L2_reg_dual = L2_reg_dual
        self.L2_reg_loss = L2_reg_loss
        self.max_opt_itr = max_opt_itr
        self.optimizer = scipy.optimize.fmin_l_bfgs_b
        self.opt_info = None
        self.discount = discount
        self.gae_lambda = gae_lambda

        self.center_adv = False
        self.positive_adv = False
        self.store_paths = False

        self._max_reps_itr = 500

        self._s_dim     = self.env._obs_dim
        self._theta_dim = self.env._obs_dim * 2 + 4

        self._plot = False

        self._sampler = BatchSampler(self)

        self._tf_sess =  tf_sess

    def init_opt(self):

        # Init dual param values
        self.param_eta = 15.
        # Adjust for linear feature vector.
        self._param_v = np.random.rand(self._theta_dim)

        # vars
         
        # self.env.new_tensor_variable(name='obs',   extra_dims=1)
        # self.env.new_tensor_variable(name='action',extra_dims=1)

        obs_var    = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='obs') 
        action_var = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='action')

        rewards    = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='rewards')
        feat_diff  = tf.placeholder(dtype=tf.float32, shape=(None, 8), name='feat_diff')
        param_v    = tf.placeholder(dtype=tf.float32, shape=(self._theta_dim, None), name='param_v')
        param_eta  = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='param_eta')


        state_info_vars = {
            k: tf.placeholder(dtype=tf.float32, shape=(None, 2), name=k) for k in self.policy.state_info_keys
        }

        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        # Policy-related symbolics
        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        dist = self.policy.distribution
        # log of the policy dist
        logli = dist.log_likelihood_sym(action_var, dist_info_vars)

        # Symbolic sample Bellman error
        delta_v = rewards + tf.matmul(feat_diff, param_v)

        # Policy loss (negative because we minimize)
        loss = - tf.reduce_mean(  tf.exp(delta_v / param_eta - tf.reduce_max(delta_v / param_eta) ))

        # Add regularization to loss.
        # reg_params = self.policy.output_layers
        # loss += self.L2_reg_loss * tf.reduce_sum( [tf.reduce_mean(tf.square(param)) for param in reg_params] ) / len(reg_params)

        # Policy loss gradient.

        loss_grad = tf.gradients(loss, self.policy.output_layer)

        input = [rewards, obs_var, feat_diff, action_var] + state_info_vars_list  + [param_eta, param_v]

        f_loss = TfFunction(inputs=input, outputs=loss)
    
        f_loss_grad = TfFunction(inputs=input, outputs=loss_grad)

        old_dist_info_vars = {
            k: tf.placeholder(dtype=tf.float32, shape=(None, 2), name='old_%s'%k) for k in dist.dist_info_keys
            }

        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        mean_kl = tf.reduce_mean(dist.kl_sym(old_dist_info_vars, dist_info_vars))

        f_kl = TfFunction(inputs=[obs_var, action_var] + state_info_vars_list + old_dist_info_vars_list, outputs=mean_kl)

        # Dual-related symbolics
        # Symbolic dual
        dual = param_eta * self.epsilon +param_eta * tf.log(
                   tf.reduce_mean(
                       tf.exp(
                           delta_v / param_eta - tf.reduce_max(delta_v / param_eta)
                       )
                   )
               ) + param_eta * tf.reduce_max(delta_v / param_eta)
        # Add L2 regularization.
        dual += self.L2_reg_dual*(tf.square(param_eta) + tf.square(1 / param_eta))

        # Symbolic dual gradient
        dual_grad = tf.gradients(dual, [param_eta, param_v])

        f_dual =  TfFunction(inputs=[rewards, feat_diff] + state_info_vars_list + [param_eta, param_v], outputs=dual)

        f_dual_grad = TfFunction(inputs=[rewards, feat_diff] + state_info_vars_list + [param_eta, param_v], outputs=dual_grad)


        self.opt_info = {
            'f_loss_grad':f_loss_grad,
            'f_loss':f_loss,
            'f_dual':f_dual,
            'f_dual_grad':f_dual_grad,
            'f_kl':f_kl
        }


    def _features(self, path):
        o = np.clip(path["observations"], -10, 10)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)


    def optimize_policy(self, itr, samples_data):
        # Init vars
        rewards = samples_data['rewards']
        actions = samples_data['actions']
        observations = samples_data['observations']

        agent_infos = samples_data["agent_infos"]
        state_info_list = []#[agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = []#[agent_infos[k] for k in self.policy.distribution.dist_info_keys]

        # Compute sample Bellman error.
        feat_diff = []
        for path in samples_data['paths']:
            feats = self._features(path)
            feats = np.vstack([feats, np.zeros(feats.shape[1])])
            feat_diff.append(feats[1:] - feats[:-1])
        
        feat_diff = np.vstack(feat_diff)

        #################
        # Optimize dual #
        #################

        # Here we need to optimize dual through BFGS in order to obtain \eta
        # value. Initialize dual function g(\theta, v). \eta > 0
        # First eval delta_v
        f_dual = self.opt_info['f_dual']
        f_dual_grad = self.opt_info['f_dual_grad']

        # Set BFGS eval function
        def eval_dual(input):
            param_eta = input[0]
            param_v = input[1:]
            params = {'param_eta':input[:1][None, :], 
                      'param_v':input[1:],
                      'rewards':rewards,
                      'feat_diff':feat_diff,
                      'state_info_list':state_info_list}
            val = f_dual(self._tf_sess, params)

            return val

        # Set BFGS gradient eval function
        def eval_dual_grad(input):
            param_eta = input[0]
            param_v = input[1:]
    
            params = {'param_eta':input[:1][None, :], 
                      'param_v':input[1:],
                      'rewards':rewards,
                      'feat_diff':feat_diff,
                      'state_info_list':state_info_list}

            grad = f_dual_grad(self._tf_sess, params)

            return np.vstack([grad[0], grad[1]]).T

        # Initial BFGS parameter values.
        x0 = np.hstack([self.param_eta, self._param_v])

        # Set parameter boundaries: \eta>0, v unrestricted.
        bounds = [(-np.inf, np.inf) for _ in x0]
        bounds[0] = (0., np.inf)

        # Optimize through BFGS
        print ('optimizing dual')
        eta_before = x0[0]
        dual_before = eval_dual(x0)
        params_ast, _, _ = self.optimizer(
            func=eval_dual, x0=x0,
            fprime=eval_dual_grad,
            bounds=bounds,
            maxiter=self.max_opt_itr,
            disp=0
        )
        dual_after = eval_dual(params_ast)

        # Optimal values have been obtained

        self.param_eta = params_ast[0]
        self._param_v = params_ast[1:]

        ###################
        # Optimize policy #
        ###################
        cur_params = self.policy.get_param_values(trainable=True)
        f_loss = self.opt_info["f_loss"]
        f_loss_grad = self.opt_info['f_loss_grad']
        input = [rewards, observations, feat_diff,
                 actions] + state_info_list  + [self.param_eta, self._param_v]

        # Set loss eval function
        def eval_loss(params):
            self.policy.set_param_values(params, trainable=True)

            params = {'param_eta':params_ast[:1][None, :], 
                      'param_v':self._param_v[:, None],
                      'rewards':rewards,
                      'feat_diff':feat_diff,
                      'observations':observations,
                      'actions':actions,
                      'state_info_list':state_info_list}
        
            val = f_loss(self._tf_sess, params)
            # val = np.random.randn(1)

            return val

        # Set loss gradient eval function
        def eval_loss_grad(params):
            self.policy.set_param_values(params, trainable=True)
            params = {'param_eta':params_ast[:1][None, :], 
                      'param_v':self._param_v[:, None],
                      'rewards':rewards,
                      'feat_diff':feat_diff,
                      'observations':observations,
                      'actions':actions,
                      'state_info_list':state_info_list}

            # grad = f_loss_grad(self._tf_sess, params)

            grad = np.array([0.])

            return grad

        loss_before = eval_loss(cur_params)
        print 'optimizing policy'
        params_ast, _, _ = self.optimizer(
            func=eval_loss, x0=cur_params,
            fprime=eval_loss_grad,
            disp=0,
            maxiter=self.max_opt_itr
        )
        loss_after = eval_loss(params_ast)

        f_kl = self.opt_info['f_kl']

        params = {'dist_info_list':dist_info_list,
          'observations':observations,
          'actions':actions,
          'state_info_list':state_info_list}

        mean_kl = f_kl(self._tf_sess, params)

        print ('loss: before %f -> %f' % (loss_before, loss_after))
        print ('eta %f -> %f' % (eta_before, self.param_eta))
        print ('mean kl ->', mean_kl)


    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            env=self.env,
        )

    def train(self):

        with self._tf_sess:

            self._tf_sess.run(tf.initialize_all_variables())

            for itr in range(self._max_reps_itr):
                print 'itr #%d | ' % itr
                paths = self._sampler.obtain_samples(itr)
                samples_data = self._sampler.process_samples(itr, paths)
                self.optimize_policy(itr, samples_data)
                print("saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)
                params["algo"] = self
                if self.store_paths:
                    params["paths"] = samples_data["paths"]
                print "saved"
                if self._plot:
                    self.update_plot()
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                                  "continue...")
