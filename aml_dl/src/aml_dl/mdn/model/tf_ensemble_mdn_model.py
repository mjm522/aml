import tensorflow as tf
import math
import numpy as np
import random
from aml_dl.utilities.tf_optimisers import optimiser_op
from aml_dl.mdn.model.tf_mdn_model import MixtureDensityNetwork
import copy


class EnsambleMDN(object):

    def __init__(self, network_params, tf_sumry_wrtr = None):

        self._device_id = network_params['device']
        self._n_ensembles = network_params['n_ensembles']
        self._dim_input = network_params['dim_input']
        self._dim_output = network_params['dim_output']
        self._mdn_ensembles = [MixtureDensityNetwork(network_params, tf_sumry_wrtr = tf_sumry_wrtr) for _ in range(self._n_ensembles)]

    def _init_model(self, input_op = None, input_tgt = None):

        with tf.device(self._device_id):

            with tf.name_scope('ensemble_input'):

                if input_op is None:
                    self._x = tf.placeholder(dtype=tf.float32, shape=[None,self._dim_input],  name="x")
                else:
                    self._x = input_op

                if input_tgt is None:
                    self._y = tf.placeholder(dtype=tf.float32, shape=[None,self._dim_output], name="y")
                else:
                    self._y = input_tgt

            for k in range(self._n_ensembles):
                with tf.name_scope('mdn_%d_'%(k,)):
                    self._mdn_ensembles[k]._init_model(input_op=self._x, input_tgt=self._y)


            with tf.name_scope('ensamble_output'):

                self._mu_ops =  [self._mdn_ensembles[k]._ops['mu'] for k in range(self._n_ensembles)]
                self._sigma_ops =  [self._mdn_ensembles[k]._ops['sigma'] for k in range(self._n_ensembles)]
                self._pi_ops =  [self._mdn_ensembles[k]._ops['pi'] for k in range(self._n_ensembles)]


        self._ops = {'x': self._x, 
                     'y': self._y,
                     'mus': self._mu_ops,
                     'sigmas': self._sigma_ops,
                     'pis': self._pi_ops}


    def get_adversarial_examples(self, data_x, data_y, loss_grad, epsilon=0.0001, no_examples=50):
    
        rand_indices = [random.randint(0,len(data_x)-1) for _ in range(no_examples)]

        x_adv = np.zeros((no_examples, self._dim_input))
        y_adv = np.zeros((no_examples, self._dim_output))

        idx = 0
        for index in rand_indices:
            x_adv[idx,:] = data_x[index,:] + epsilon*np.sign(loss_grad[index])
            y_adv[idx,:] = data_y[index]
            idx += 1

        return x_adv, y_adv

    def train(self, x_train , y_train , sess, iterations):
        #training session
        sess.run(tf.global_variables_initializer())

        with tf.device(self._device_id):
            # Keeping track of loss progress as we train
            loss = np.zeros([self._n_ensembles, iterations])
            
            for i in range(iterations):
                #train them parrallely
                for k in range(self._n_ensembles):
                    train_op = self._mdn_ensembles[k]._ops['train']
                    loss_op  = self._mdn_ensembles[k]._ops['loss']
                    grad_op  = self._mdn_ensembles[k]._ops['loss_grad']
                    
                    #compute value of the gradients
                    loss_grad = sess.run(grad_op,feed_dict={self._mdn_ensembles[k]._ops['x']: x_train, self._mdn_ensembles[k]._ops['y']: y_train})
                    
                    #get adversarial examples
                    x_adv, y_adv = self.get_adversarial_examples(data_x = x_train, data_y = y_train, epsilon=0.000001, loss_grad=loss_grad, no_examples=3)
                    
                    x_train = copy.deepcopy(np.append(x_train, x_adv, axis=0))
                    y_train = copy.deepcopy(np.append(y_train, y_adv, axis=0))

                    _, loss[k,i] = sess.run([train_op, loss_op], feed_dict={self._mdn_ensembles[k]._ops['x']: x_train, self._mdn_ensembles[k]._ops['y']: y_train})


            return loss

    def run_op(self, sess, op,  xs, ys = None):
        out = []

        if ys is None:
            out = sess.run(self._ops[op], feed_dict = { self._x: xs })
        else:
            out = sess.run(self._ops[op], feed_dict = { self._x: xs, self._y: ys })


        return out

    def forward(self, sess, xs):

        mean_out = np.zeros((len(xs),self._dim_output))
        var_out = np.zeros((len(xs),self._dim_output))

        for model in self._mdn_ensembles:

            mu, sigma, pi = model.forward(sess, xs)

            mu = np.reshape(mu,(-1,self._dim_output))

            # Correct only for the single kernel MDN case
            mean_out += mu
            var_out += sigma + np.square(mu)

        mean_out /= len(self._mdn_ensembles)
        var_out /= len(self._mdn_ensembles)
        var_out -= np.square(mean_out)

        return mean_out, var_out

