import numpy as np
import scipy.optimize
import tensorflow as tf
from sampler import BatchSampler
from utilities import TfFunction
from aml_dl.utilities.tf_optimisers import optimiser_op

class REPS(object):

    def __init__(self, env, policy, tf_sess, config):
        """
        :param epsilon: Max KL divergence between new policy and old policy.
        :param L2_reg_dual: Dual regularization
        :param L2_reg_loss: Loss regularization
        :param max_opt_itr: Maximum number of batch optimization iterations.
        :param optimizer: Module path to the optimizer. It must support the same interface as
        scipy.optimize.fmin_l_bfgs_b.
        :return:
        """
        
        self._env     = env
        self._policy  = policy
        self._tf_sess =  tf_sess

        self._discount     = config['discount']
        self._gae_lambda   = config['gae_lambda']
        self._epsilon      = config['epsilon']
        self._L2_reg_dual  = config['L2_reg_dual']
        self._L2_reg_loss  = config['L2_reg_loss']
        self._max_opt_itr  = config['max_opt_itr']
        self._positive_adv = config['positive_adv']
        self._center_adv   = config['center_adv']
        self._tf_opt_params= config['tf_opt_params']

        self._optimizer = scipy.optimize.fmin_l_bfgs_b
        self._opt_info   = None

        self._max_reps_itr = 500

        self._state_dim = self._env._state_dim
        self._action_dim = self._env._action_dim

        self._theta_dim = config['feature_dim']

        self._sampler = BatchSampler(self)

        # initial legrange multiplier.
        self._param_eta   = 15.
        # initial linear feature vector.
        self._param_theta = np.random.rand(self._theta_dim)

        self._plot = False
        self._store_paths = False

        
    def init_opt(self):

        '''
        this function builds the computational graph required for the reps algorithm
        by building a computational graph, it becomes easy to train the neural network
        '''

        #create some placeholders for essential variables
        obs_var     = tf.placeholder(dtype=tf.float32, shape=(None, self._state_dim), name='obs') 
        action_var  = tf.placeholder(dtype=tf.float32, shape=(None, self._action_dim), name='action')
        rewards     = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='rewards')
        feat_diff   = tf.placeholder(dtype=tf.float32, shape=(None, self._theta_dim), name='feat_diff')
        param_theta = tf.placeholder(dtype=tf.float32, shape=(self._theta_dim, None), name='param_theta')
        param_eta   = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='param_eta')

        # symbolic policy mean and covariance
        dist_vars = self._policy.dist_vars_sym(obs_var)

        # symbolic log likelihood of the policy dist for current actions
        logli = self._policy.log_likelihood_sym(action_var, dist_vars)

        # symbolic sample bellman error
        delta_v = rewards + tf.matmul(feat_diff, param_theta)

        # symbolic policy loss (negative because we minimize)
        loss = - tf.reduce_mean(logli*tf.exp(delta_v / param_eta - tf.reduce_max(delta_v / param_eta) ))

        # Add regularization to loss.
        # reg_params = self.policy.output_layers
        # loss += self.L2_reg_loss * tf.reduce_sum( [tf.reduce_mean(tf.square(param)) for param in reg_params] ) / len(reg_params)

        # symbolic policy loss gradient.
        loss_grad = tf.gradients(loss, self._policy.output_layer)

        #create a symbolic placeholder to store older variables
        old_dist_vars = {
            k: tf.placeholder(dtype=tf.float32, shape=(None, self._action_dim), name='old_%s'%k) for k in self._policy.dist_keys
            }

        old_dist_vars_list = [old_dist_vars[k] for k in self._policy.dist_keys]

        
        #symbolic kl divergence of the old policy and new policy
        mean_kl = tf.reduce_mean(self._policy.kl_sym(old_dist_vars, dist_vars))

        # Dual-related symbolics
        # symbolic dual expression
        dual = param_eta * self._epsilon +param_eta * tf.log(
                   tf.reduce_mean(
                       tf.exp(
                           delta_v / param_eta - tf.reduce_max(delta_v / param_eta)
                       )
                   )
               ) + param_eta * tf.reduce_max(delta_v / param_eta)

        # add L2 regularization.
        dual += self._L2_reg_dual*(tf.square(param_eta) + tf.square(1 / param_eta))

        # symbolic dual gradient with respect to eta and v
        dual_grad = tf.gradients(dual, [param_eta, param_theta])

        
        #symbolic loss function 
        f_loss = TfFunction(inputs=[rewards, obs_var, feat_diff, action_var] + [param_eta, param_theta], 
                            outputs=loss)

        #symbolic loss function gradient
        f_loss_grad = TfFunction(inputs=[rewards, obs_var, feat_diff, action_var] + [param_eta, param_theta], 
                                 outputs=loss_grad)


        #symbolic kl divergence function wrapper
        f_kl = TfFunction(inputs=[obs_var, action_var] + old_dist_vars_list, 
                          outputs=mean_kl)

        #symbolic dual function wrapper
        f_dual =  TfFunction(inputs=[rewards, feat_diff]  + [param_eta, param_theta], 
                             outputs=dual)

        #symbolic dual gradient function wrapper
        f_dual_grad = TfFunction(inputs=[rewards, feat_diff] + [param_eta, param_theta], 
                                outputs=dual_grad)

        #tf policy optimizer
        train_op = optimiser_op(loss, self._tf_opt_params)

        self._opt_info = {
            'f_loss':f_loss,
            'f_loss_grad':f_loss_grad,
            'f_dual':f_dual,
            'f_dual_grad':f_dual_grad,
            'f_kl':f_kl,
            'train_op':train_op,
        }


    def _features(self, path):
        o = np.clip(path["observations"], -10, 10)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)


    def optimize_policy(self, itr, samples_data):
        # Init vars
        rewards       = samples_data['rewards']
        actions       = samples_data['actions']
        observations  = samples_data['observations']
        policy_vars   = samples_data["policy_vars"]
        features      = samples_data["features"]

        policy_out    = np.hstack([policy_vars['mean'], policy_vars['log_std']])


        f_kl        = self._opt_info['f_kl']
        f_dual      = self._opt_info['f_dual']
        f_dual_grad = self._opt_info['f_dual_grad']
        f_loss      = self._opt_info["f_loss"]
        f_loss_grad = self._opt_info['f_loss_grad']

        policy_vars_list = [policy_vars[k] for k in self._policy.dist_keys]

        # compute actual sample Bellman error using numerical data.
        feat_diff = []
        for path in samples_data['paths']:
            feats = self._features(path)
            feats = np.vstack([feats, np.zeros(feats.shape[1])])
            feat_diff.append(feats[1:] - feats[:-1])
        
        feat_diff = np.vstack(feat_diff)

        #after computing the feature difference, now it is time to compute the numerical values
        #using the symbolic expressions created above

        #################
        # Optimize dual #
        #################

        # Here we need to optimize dual through BFGS in order to obtain \eta
        # value. Initialize dual function g(\theta, v). \eta > 0
        # First eval delta_v
        

        # create the dual eval function from the symbolic function
        def eval_dual(param_vec):

            f_params = {'param_eta':param_vec[:1][None, :], 
                      'param_theta':param_vec[1:],
                      'rewards':rewards,
                      'feat_diff':feat_diff}

            val = f_dual(self._tf_sess, f_params)

            return val

        # create dual gradient eval function from symbolic function
        def eval_dual_grad(param_vec):
    
            f_params = {'param_eta':param_vec[:1][None, :], 
                      'param_theta':param_vec[1:],
                      'rewards':rewards,
                      'feat_diff':feat_diff}

            grad = f_dual_grad(self._tf_sess, f_params)

            return np.vstack([grad[0], grad[1]]).T

        # Initial BFGS parameter values.
        x0 = np.hstack([self._param_eta, self._param_theta])

        # Set parameter boundaries: \eta>0, v unrestricted.
        bounds = [(-np.inf, np.inf) for _ in x0]
        bounds[0] = (0., np.inf)

        # Optimize through BFGS
        print ('optimizing dual')
        eta_before = x0[0]

        dual_before = eval_dual(x0)

        #optimize using BFGS optimizer
        params_opt, _, _ = self._optimizer(
            func=eval_dual, #dual function to be optimized
            x0=x0, #initial values
            fprime=eval_dual_grad, #dual function gradient 
            bounds=bounds, #bounds of the values
            maxiter=self._max_opt_itr, #maximum number of BFGS iterations
            disp=0
        )

        dual_after = eval_dual(params_opt)

        # Optimal values have been obtained
        self._param_eta   = params_opt[0]
        self._param_theta = params_opt[1:]


        ###################
        # Optimize policy #
        ###################

        # mean_kl = [0] #f_kl(self._tf_sess, params)

        self._tf_sess.run(self._opt_info['train_op'], feed_dict={'rewards:0':rewards[:, None], 
                                                                 'feat_diff:0':feat_diff,
                                                                  'param_eta:0':params_opt[:1][None, :],
                                                                  'param_theta:0':self._param_theta[:, None],
                                                                   'x:0':features,
                                                                    'action:0':actions,})

        print ('eta %f  -> %f' % (eta_before, self._param_eta))
        print ('dual %f -> %f' % (dual_before[0], dual_after[0]))
        # print ('kl      -> %f' % mean_kl[0])


    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self._policy,
            env=self._env,
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
                
                if self._store_paths:
                    params["paths"] = samples_data["paths"]
                    print "saved"

                if self._plot:
                    self.update_plot()
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                                  "continue...")

    def update_plot(self):
        pass
