import tensorflow as tf
import math
import numpy as np

NHIDDEN = 24
STDEV = 0.5
KMIX = 24 # number of mixtures
NOUT = KMIX * 3 # pi, mu, stdev


def get_mixture_parameters(output, dim_output = 1, n_kernels = KMIX):
  
  out_pi = tf.placeholder(dtype=tf.float32, shape=[None,n_kernels], name="mixparam")
  out_sigma = tf.placeholder(dtype=tf.float32, shape=[None,n_kernels], name="mixparam")
  out_mu = tf.placeholder(dtype=tf.float32, shape=[None,n_kernels*dim_output], name="mixparam")

  out_pi, out_sigma, out_mu = tf.split_v(output, [n_kernels,n_kernels,n_kernels*dim_output], 1)

  max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
  out_pi = tf.sub(out_pi, max_pi)

  out_pi = tf.exp(out_pi)

  normalize_pi = 1./(tf.reduce_sum(out_pi, 1, keep_dims=True))
  
  out_pi = tf.mul(normalize_pi, out_pi)

  out_sigma = tf.exp(out_sigma)

  return out_pi, out_sigma, out_mu


def tf_simple_mdn_model(dim_input = 1, dim_output = NOUT, stddev = 0.5, n_hidden = 24):

  x = tf.placeholder(dtype=tf.float32, shape=[None,dim_input], name="x")
  y = tf.placeholder(dtype=tf.float32, shape=[None,1], name="y")

  Wh = tf.Variable(tf.random_normal([dim_input,n_hidden], stddev=stddev, dtype=tf.float32))
  bh = tf.Variable(tf.random_normal([1,n_hidden], stddev=stddev, dtype=tf.float32))

  Wo = tf.Variable(tf.random_normal([n_hidden,dim_output], stddev=stddev, dtype=tf.float32))
  bo = tf.Variable(tf.random_normal([1,dim_output], stddev=stddev, dtype=tf.float32))

  hidden_layer = tf.nn.tanh(tf.matmul(x, Wh) + bh)
  output = tf.matmul(hidden_layer,Wo) + bo

  out_pi, out_sigma, out_mu = get_mixture_parameters(output)

  loss = get_loss(out_pi, out_sigma, out_mu,y)

  return out_pi, out_sigma, out_mu, loss, x, y


def tf_general_mdn_model(dim_input = 12, dim_output = 1, n_hidden = 24, n_kernels = 2, stddev = 0.5):

  # 3 parameters: pi, mu, stdev
  params_per_output_dim = (2 + dim_output)
  n_params_out = n_kernels*params_per_output_dim

  x = tf.placeholder(dtype=tf.float32, shape=[None,dim_input], name="x")
  y = tf.placeholder(dtype=tf.float32, shape=[None,1], name="y")

  Wh = tf.Variable(tf.random_normal([dim_input,n_hidden], stddev=stddev, dtype=tf.float32))
  bh = tf.Variable(tf.random_normal([1,n_hidden], stddev=stddev, dtype=tf.float32))

  Wo = tf.Variable(tf.random_normal([n_hidden,n_params_out], stddev=stddev, dtype=tf.float32))
  bo = tf.Variable(tf.random_normal([1,n_params_out], stddev=stddev, dtype=tf.float32))

  hidden_layer = tf.nn.tanh(tf.matmul(x, Wh) + bh)
  output = tf.matmul(hidden_layer,Wo) + bo

  out_pi, out_sigma, out_mu = get_mixture_parameters(output=output, dim_output = dim_output, n_kernels=n_kernels)

  loss = get_loss(out_pi, out_sigma, out_mu, y)

  train_op = get_train(loss)

  output_ops = {'pi': out_pi, 'sigma': out_sigma, 'mu': out_mu, 
                'loss': loss, 'z_hidden': hidden_layer, 'train': train_op, 'x': x, 'y': y}

  return output_ops


def tf_normal(y, mu, sigma):

  oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi) # normalisation factor for gaussian, not needed.
  result = tf.sub(y, mu)
  result = tf.mul(result,1./(sigma))
  result = -tf.square(result)/2
  return tf.mul(tf.exp(result),1./(sigma))*oneDivSqrtTwoPI

def get_loss(out_pi, out_sigma, out_mu, y):

  result = tf_normal(y, out_mu, out_sigma)
  result = tf.mul(result, out_pi)
  result = tf.reduce_sum(result, 1, keep_dims=True)
  result = -tf.log(result)

  return tf.reduce_mean(-result)


def get_train(loss_op):

  train_op = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss_op)

  return train_op


def tf_pushing_model(dim_input = 12, dim_output = 1, n_hidden = 24, n_kernels = 2, stddev = 0.5):


  output_ops = tf_general_mdn_model(dim_input = dim_input, dim_output = dim_output, 
                                    n_hidden = n_hidden, n_kernels = n_kernels, stddev = stddev)

  return output_ops


#quadratic loss function
def get_loss_fwd(output, target):

  return tf.sqrt(tf.reduce_mean(tf.square(tf.sub(target, output))))


def tf_fwd_pushing_model(dim_input=7, dim_output=7, n_hidden_layers=3, units_in_hidden_layers=None, stddev=0.5):

  if units_in_hidden_layers is None:

    units_in_hidden_layers = [dim_input for _ in range(n_hidden_layers)]

    units_in_hidden_layers[n_hidden_layers-1] = dim_output

  else:

    if len(units_in_hidden_layers) != n_hidden_layers:
      print "param: units_in_hidden_layers should be equal to param: n_hidden_layers"

  x = tf.placeholder(dtype=tf.float32, shape=[None, dim_input],  name="x")
  y = tf.placeholder(dtype=tf.float32, shape=[None, dim_output], name="y")

  for h in range(n_hidden_layers):

    if h == 0:
      #input layer
      Wi = tf.Variable(tf.random_normal([dim_input, units_in_hidden_layers[h]], stddev=stddev, dtype=tf.float32))
      bi = tf.Variable(tf.random_normal([1, units_in_hidden_layers[h]], stddev=stddev, dtype=tf.float32))

      hidden_layer = tf.nn.tanh(tf.matmul(x, Wi) + bi)

    else:
      #all other hidden layers
      Wh = tf.Variable(tf.random_normal([units_in_hidden_layers[h-1], units_in_hidden_layers[h]], stddev=stddev, dtype=tf.float32))
      bh = tf.Variable(tf.random_normal([1,units_in_hidden_layers[h]], stddev=stddev, dtype=tf.float32))

      hidden_layer = tf.nn.tanh(tf.matmul(hidden_layer, Wh) + bh)

  #ouput layer
  Wo = tf.Variable(tf.random_normal([units_in_hidden_layers[-1], dim_output], stddev=stddev, dtype=tf.float32))
  bo = tf.Variable(tf.random_normal([1, dim_output], stddev=stddev, dtype=tf.float32))

  output = tf.matmul(hidden_layer, Wo) + bo

  loss = get_loss_fwd(output, y)

  train_op = get_train(loss)

  output_ops = {'output' : output, 'loss': loss, 'z_hidden': hidden_layer, 'train': train_op, 'x': x, 'y': y}

  return output_ops