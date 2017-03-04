


import tensorflow as tf
import math
import numpy as np


class MixtureDensityNetwork(object):

  def __init__(self, dim_input = 1, dim_output = 1, n_kernels = 5, n_hidden = 24):
    self._dim_input = dim_input
    self._dim_output = dim_output
    self._n_kernels = n_kernels
    self._n_hidden = n_hidden


  def _init_model(self):

    self._x = tf.placeholder(dtype=tf.float32, shape=[None,self._dim_input], name="x")
    self._y = tf.placeholder(dtype=tf.float32, shape=[None,self._dim_output], name="y")

    self._output_fc_op = self._init_fc_layer(self._x)

    self._mus_op, self._sigmas_op, self._pis_op = self._init_mixture_parameters(self._output_fc_op)


    self._loss_op = self._init_loss(self._mus_op, self._sigmas_op, self._pis_op, self._y)


    self._train_op = self._init_train(self._loss_op)


    self._ops = {'x': self._x, 'y': self._y, 'mu': self._mus_op, 'sigma': self._sigmas_op, 'pi': self._pis_op, 'loss': self._loss_op, 'train': self._train_op}




  def _init_fc_layer(self, input, stddev = 0.1):

    n_params_out = (self._dim_output + 2)*self._n_kernels

    Wh = tf.Variable(tf.random_normal([self._dim_input, self._n_hidden], stddev=stddev, dtype=tf.float32))
    bh = tf.Variable(tf.random_normal([1, self._n_hidden], stddev=stddev, dtype=tf.float32))

    Wo = tf.Variable(tf.random_normal([self._n_hidden, n_params_out], stddev=stddev, dtype=tf.float32))
    bo = tf.Variable(tf.random_normal([1, n_params_out], stddev=stddev, dtype=tf.float32))

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

    train_op = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss_op)

    return train_op


  # Do the log trick here if it is not good enough the way it is now
  def _kernels(self, mus, sigmas, ys):
    c = self._dim_output
    m = self._n_kernels

    reshaped_ys = tf.reshape(ys, [-1, c, 1])
    reshaped_sigmas = tf.reshape(sigmas, [-1, 1, m])

    diffs = tf.sub(mus, reshaped_ys) # broadcasting
    expoents = tf.reduce_sum( tf.multiply(diffs,diffs), 1, keep_dims=True )

    sigmacs = tf.pow(reshaped_sigmas,c)

    expoents = tf.multiply(-0.5,tf.multiply(tf.reciprocal(sigmacs), expoents))

    denominators = tf.pow(2*np.pi,c/2.0)*tf.sqrt(sigmacs)
    
    out = tf.div(tf.exp(expoents),denominators)

    return out


  def run(self, sess, xs, ys = None):
    out = []
    if ys is None:
      out = sess.run([self._mus_op, self._sigmas_op, self._pis_op], feed_dict = { self._x: xs })
    else:
      out = sess.run([self._mus_op, self._sigmas_op, self._pis_op, self._loss_op], feed_dict = { self._x: xs, self._y: ys})

    return out

  def run_op(self, sess, op,  xs, ys = None):
    out = []

    if ys is None:
      out = sess.run([self._ops[op]], feed_dict = { self._x: xs })
    else:
      out = sess.run([self._ops[op]], feed_dict = { self._x: xs, self._y: ys })


    return out






# class MixtureOfGaussians(object):


#   def sample_pi_idx(self, x, pdf):
#     N = pdf.size
#     acc = 0

#     for i in range(0, N):
#       acc += pdf[i]
#       if (acc >= x):
#         return i

#         print 'failed to sample mixture weight index'


#     return -1


#   def max_pi_idx(self, pdf):

#     i = np.argmax(pdf)

#     return i

#   def sample_gaussian(self, rn, mu, std):


#     return mu + rn*std

#   def generate_mixture_samples_from_max_pi(self, out_pi, out_mu, out_sigma, m_samples=10):


#     # Number of test inputs

#     N = out_mu.shape[0]

#     M = m_samples

#     result = np.random.rand(N, M)
#     rn  = np.random.randn(N, M) # normal random matrix (0.0, 1.0)

#     # Generates M samples from the mixture for each test input

#     for j in range(M):
#       for i in range(0, N):
#         idx = self.max_pi_idx(out_pi[i])
#         mu = out_mu[i, idx]
#         std = out_sigma[i, idx]
#         result[i, j] = self.sample_gaussian(rn[i, j], mu, std)


#     return result


#   def generate_mixture_samples(self, out_pi, out_mu, out_sigma, m_samples=10):


#     # Number of test inputs
#     N = out_mu.shape[0]
#     M = m_samples

#     result = np.random.rand(N, M) # initially random [0, 1]
#     rn  = np.random.randn(N, M) # normal random matrix (0.0, 1.0)
#     mu  = 0
#     std = 0
#     idx = 0

#     # Generates M samples from the mixture for each test input
#     for j in range(M):
#       for i in range(0, N):
#         idx = self.sample_pi_idx(result[i, j], out_pi[i])
#         mu = out_mu[i, idx]
#         std = out_sigma[i, idx]
#         result[i, j] = self.sample_gaussian(rn[i, j], mu, std)


#     return result




