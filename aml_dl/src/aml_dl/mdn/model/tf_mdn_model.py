import tensorflow as tf
import math
import numpy as np
from aml_dl.utilities.tf_optimisers import optimiser_op


class MixtureDensityNetwork(object):

    def __init__(self, network_params, tf_sumry_wrtr=None):
      
        self._dim_input = network_params['dim_input']
        self._dim_output = network_params['dim_output']
        self._n_kernels = network_params['k_mixtures']
        self._n_hidden = network_params['n_hidden']
        self._optimiser = network_params['optimiser']
        self._tf_sumry_wrtr = tf_sumry_wrtr

    def _init_model(self, input_op = None, input_tgt = None):

        with tf.name_scope('input'):

            if input_op is None:
                self._x = tf.placeholder(dtype=tf.float32, shape=[None,self._dim_input],  name="x")
            else:
                self._x = input_op

            if input_tgt is None:
                self._y = tf.placeholder(dtype=tf.float32, shape=[None,self._dim_output], name="y")
            else:
                self._y = input_tgt

        with tf.name_scope('output_fc_op'):
            self._output_fc_op = self._init_fc_layer(self._x)
            if self._tf_sumry_wrtr is not None:
                self._tf_sumry_wrtr.add_variable_summaries(self._output_fc_op)

        with tf.name_scope('init_mixture_params'):
            self._mus_op, self._sigmas_op, self._pis_op = self._init_mixture_parameters(self._output_fc_op)
            if self._tf_sumry_wrtr is not None:
                self._tf_sumry_wrtr.add_variable_summaries(self._mus_op)
                self._tf_sumry_wrtr.add_variable_summaries(self._sigmas_op)
                self._tf_sumry_wrtr.add_variable_summaries(self._pis_op)

        with tf.name_scope('cost_inv'):
            self._loss_op = self._init_loss(self._mus_op, self._sigmas_op, self._pis_op, self._y)
            #gradient of the loss function with respect to the input
            self._loss_grad = tf.gradients(self._loss_op, [self._x])[0]
            if self._tf_sumry_wrtr is not None:
                self._tf_sumry_wrtr.add_variable_summaries(self._loss_op)

      
        self._train_op = self._init_train(self._loss_op)
        self._ops = {'x': self._x, 
                     'y': self._y, 
                     'mu': self._mus_op, 
                     'sigma': self._sigmas_op, 
                     'pi': self._pis_op, 
                     'loss': self._loss_op, 
                     'train': self._train_op,
                     'loss_grad':self._loss_grad}

        if self._tf_sumry_wrtr is not None:
            self._tf_sumry_wrtr.write_summary()

    def _init_fc_layer(self, input, stddev = 0.5):

        n_params_out = (self._dim_output + 2)*self._n_kernels

        input_op = input

        if type(self._n_hidden) == type([]):



            for i in range(0,len(self._n_hidden)):

                input_dim = input_op.get_shape().dims[1].value

                Wh = tf.Variable(tf.random_normal([input_dim, self._n_hidden[i]], stddev=stddev, dtype=tf.float32), name='w_' + str(i))
                bh = tf.Variable(tf.random_normal([1, self._n_hidden[i]], stddev=stddev, dtype=tf.float32), name='b_' + str(i))

                input_op = tf.nn.tanh(tf.matmul(input_op, Wh) + bh)

            input_dim = input_op.get_shape().dims[1].value

            Wo = tf.Variable(tf.random_normal([input_dim, n_params_out], stddev=stddev, dtype=tf.float32), name='w_out_fc')
            bo = tf.Variable(tf.random_normal([1, n_params_out], stddev=stddev, dtype=tf.float32), name='b_out_fc')

            output_fc = tf.matmul(input_op, Wo) + bo

            return output_fc

        else:

            Wh = tf.Variable(tf.random_normal([self._dim_input, self._n_hidden], stddev=stddev, dtype=tf.float32), name='w_0')
            bh = tf.Variable(tf.random_normal([1, self._n_hidden], stddev=stddev, dtype=tf.float32), name='b_0')

            Wo = tf.Variable(tf.random_normal([self._n_hidden, n_params_out], stddev=stddev, dtype=tf.float32), name='w_1')
            bo = tf.Variable(tf.random_normal([1, n_params_out], stddev=stddev, dtype=tf.float32), name='b_1')

            hidden_layer = tf.nn.tanh(tf.matmul(input, Wh) + bh)
            output_fc = tf.matmul(hidden_layer, Wo) + bo

            return output_fc

    def _init_mixture_parameters(self, output):

        c = self._dim_output
        m = self._n_kernels

        reshaped_output = tf.reshape(output,[-1, (c+2), m])
        mus = reshaped_output[:, :c, :]
        sigmas = tf.exp(reshaped_output[:, c, :])
        pis = tf.nn.softmax(reshaped_output[:, c+1, :])

        return mus, sigmas, pis


    def _init_loss(self, mus, sigmas, pis, ys):

        m = self._n_kernels

        kernels = self._kernels(mus, sigmas, ys)

        result = tf.multiply(kernels,tf.reshape(pis, [-1, 1, m]))
        result = tf.reduce_sum(result, 2, keep_dims=True)

        epsilon = 1e-20
        result = -tf.log(tf.maximum(result,1e-20))


        return tf.reduce_mean(result, 0)

    def _init_train(self,loss_op):

        train_op = optimiser_op(loss_op, self._optimiser)

        return train_op


    # Do the log trick here if it is not good enough the way it is now
    def _kernels(self, mus, sigmas, ys):
        c = self._dim_output
        m = self._n_kernels

        reshaped_ys = tf.reshape(ys, [-1, c, 1])
        reshaped_sigmas = tf.reshape(sigmas, [-1, 1, m])

        diffs = tf.subtract(mus, reshaped_ys) # broadcasting
        expoents = tf.reduce_sum( tf.multiply(diffs,diffs), 1, keep_dims=True )

        sigmacs = tf.pow(reshaped_sigmas,c)

        expoents = tf.multiply(-0.5,tf.multiply(tf.reciprocal(sigmacs), expoents))

        denominators = tf.pow(2*np.pi,c/2.0)*tf.sqrt(sigmacs)

        out = tf.div(tf.exp(expoents),denominators)

        return out

    def _mixture(self, kernels, pis):

        result = tf.multiply(kernels,tf.reshape(pis, [-1, 1, m]))
        mixture = tf.reduce_sum(result, 2, keep_dims=True)



    def run(self, sess, xs, ys = None):
        out = []
        if ys is None:
            out = sess.run([self._mus_op, self._sigmas_op, self._pis_op], feed_dict = { self._x: xs })
        else:
            out = sess.run([self._mus_op, self._sigmas_op, self._pis_op, self._loss_op], feed_dict = { self._x: xs, self._y: ys})

        return out

    def forward(self, sess, xs):

        out = sess.run([self._mus_op, self._sigmas_op, self._pis_op], feed_dict = { self._x: xs })

        return out


    def run_op(self, sess, op,  xs, ys = None):
        out = []

        if ys is None:
            out = sess.run([self._ops[op]], feed_dict = { self._x: xs })
        else:
            out = sess.run([self._ops[op]], feed_dict = { self._x: xs, self._y: ys })


        return out

