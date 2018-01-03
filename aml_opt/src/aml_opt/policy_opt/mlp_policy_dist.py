import numpy as np
import tensorflow as tf

def get_activation(key):
    if key == 'tanh':
        return tf.nn.tanh
    elif key == 'relu':
        return tf.nn.relu
    else:
        return tf.nn.relu


class MLPPolicyDist(object):

    def __init__(self, output_dim, hidden_sizes, hidden_activation, 
        mu_activation, sig_activation, input_shape, batch_norm=False, input_layer=None):
        
        self._out_dim =  output_dim

        self._x = tf.placeholder(dtype=tf.float32, shape=[None, input_shape], name="x")
        
        #first layer
        if input_layer is None:
            weights = tf.Variable(tf.random_normal([input_shape, hidden_sizes[0]]))
            bias    = tf.Variable(tf.random_normal([hidden_sizes[0]]))
            l_in    = get_activation(hidden_activation)(tf.add(tf.matmul(self._x, weights), bias))
        else:
            l_in = input_layer
        
        self._layers = [l_in]

        #intermediate layers
        l_hid = l_in
        for idx, hidden_size in enumerate(hidden_sizes):

            weights = tf.Variable(tf.random_normal([l_hid.get_shape()[-1].value, hidden_size]))
            bias    = tf.Variable(tf.random_normal([hidden_size]))

            l_hid   = get_activation(hidden_activation)(tf.add(tf.matmul(l_hid, weights), bias))

            if batch_norm:
                l_hid = tf.contrib.layers.batch_norm(l_hid, center=True, scale=True)

            self._layers.append(l_hid)

        weights_mu = tf.Variable(tf.random_normal([hidden_sizes[-1], output_dim]))
        bias_mu    = tf.Variable(tf.random_normal([output_dim]))

        weights_sig = tf.Variable(tf.random_normal([hidden_sizes[-1], output_dim]))
        bias_sig    = tf.Variable(tf.random_normal([output_dim]))

        l_mu    = get_activation(mu_activation)(tf.add(tf.matmul(l_hid, weights_mu), bias_mu))
        l_sig   = get_activation(sig_activation)(tf.add(tf.matmul(l_hid, weights_sig), bias_sig))

        self._l_out   = tf.concat([l_mu, l_sig], 1)

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output

    def get_output(self, sess, x):

        result  = sess.run([self._l_out], feed_dict = {self._x: x})

        return  result[0]
