import tensorflow as tf
import math

NHIDDEN = 24
STDEV = 0.5
KMIX = 24 # number of mixtures
NOUT = KMIX * 3 # pi, mu, stdev


def get_mixture_coef(output):
  out_pi = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")
  out_sigma = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")
  out_mu = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")

  out_pi, out_sigma, out_mu = tf.split(1, 3, output)

  max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
  out_pi = tf.sub(out_pi, max_pi)

  out_pi = tf.exp(out_pi)

  normalize_pi = tf.inv(tf.reduce_sum(out_pi, 1, keep_dims=True))
  out_pi = tf.mul(normalize_pi, out_pi)

  out_sigma = tf.exp(out_sigma)

  return out_pi, out_sigma, out_mu


def tf_normal(y, mu, sigma):

  oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi) # normalisation factor for gaussian, not needed.
  result = tf.sub(y, mu)
  result = tf.mul(result,tf.inv(sigma))
  result = -tf.square(result)/2
  return tf.mul(tf.exp(result),tf.inv(sigma))*oneDivSqrtTwoPI

def get_loss(out_pi, out_sigma, out_mu, y):

  result = tf_normal(y, out_mu, out_sigma)
  result = tf.mul(result, out_pi)
  result = tf.reduce_sum(result, 1, keep_dims=True)
  result = -tf.log(result)

  return tf.reduce_mean(result)


def get_train(loss_op):

  train_op = tf.train.AdamOptimizer().minimize(loss_op)

  return train_op


def simple_mdn_model(dim_input = 1):

  x = tf.placeholder(dtype=tf.float32, shape=[None,dim_input], name="x")
  y = tf.placeholder(dtype=tf.float32, shape=[None,1], name="y")

  Wh = tf.Variable(tf.random_normal([dim_input,NHIDDEN], stddev=STDEV, dtype=tf.float32))
  bh = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=STDEV, dtype=tf.float32))

  Wo = tf.Variable(tf.random_normal([NHIDDEN,NOUT], stddev=STDEV, dtype=tf.float32))
  bo = tf.Variable(tf.random_normal([1,NOUT], stddev=STDEV, dtype=tf.float32))

  hidden_layer = tf.nn.tanh(tf.matmul(x, Wh) + bh)
  output = tf.matmul(hidden_layer,Wo) + bo

  out_pi, out_sigma, out_mu = get_mixture_coef(output)

  loss = get_loss(out_pi, out_sigma, out_mu,y)

  return out_pi, out_sigma, out_mu, loss, x, y


