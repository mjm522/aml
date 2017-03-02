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
    out_mu = tf.placeholder(dtype=tf.float32, shape=[None,n_kernels], name="mixparam")

    out_pi, out_sigma, out_mu = tf.split(1, 3, output)

    max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
    out_pi = tf.sub(out_pi, max_pi)

    out_pi = tf.exp(out_pi)

    normalize_pi = 1./(tf.reduce_sum(out_pi, 1, keep_dims=True))

    out_pi = tf.mul(normalize_pi, out_pi)

    out_sigma = tf.exp(out_sigma)

    return out_pi, out_sigma, out_mu

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
    n_params_out = n_kernels*3 #*dim_output

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


def tf_pushing_model(dim_input = 12, dim_output = 1, n_hidden = 24, n_kernels = 2, stddev = 0.5):


    output_ops = tf_general_mdn_model(dim_input = dim_input, dim_output = dim_output, 
                                    n_hidden = n_hidden, n_kernels = n_kernels, stddev = stddev)

    return output_ops





def create_conv_layer(data_in, num_inp_channels, filter_size,  num_filters, strides=None, padding='SAME', stddev=0.05, max_pooling=None):
    # Shape of the filter-weights for the convolution.
    shape    = [filter_size, filter_size, num_inp_channels, num_filters]
    # Create new weights aka. filters with the given shape.
    weights  = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    # Create new biases, one for each filter.
    biases   = tf.Variable(tf.constant(stddev, shape=[num_filters]))

    if strides is None:
        strides = [1,1,1,1]
    # Create the TensorFlow operation for convolution.
    layer    = tf.nn.conv2d(input=data_in, filter=weights, strides=strides, padding=padding)
    # Add the biases to the results of the convolution.
    layer   += biases

    # Use pooling to down-sample the image resolution
    if max_pooling is not None:
      pool_ksize   = [1, max_pooling['x'], max_pooling['y'], 1]
      pool_strides = [1, max_pooling['x'], max_pooling['y'], 1]
      layer = tf.nn.max_pool(value=layer, ksize=pool_ksize, strides=pool_strides, padding=padding)

    # Rectified Linear Unit (ReLU).
    layer = tf.nn.relu(layer)

    return layer, weights

def create_fc_layer(data_in, num_inputs, num_outputs, stddev=0.05, use_relu=True):
    # Create new weights and biases.
    weights  = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=stddev))
    biases   = tf.Variable(tf.constant(stddev, shape=[num_outputs]))
    # Calculate the layer 
    layer = tf.matmul(data_in, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer
    

########################################################FORWARD MODEL######################################
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
          raise ValueError

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

    output_ops = {'output' : output, 'loss': loss, 'last_hidden': hidden_layer, 'train': train_op, 'x': x, 'y': y}

    return output_ops

##############################################################CNN LAYERS##########################################

def get_loss_cnn(output, target):
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target)
    # cost = tf.reduce_mean(cross_entropy)
    cost =  tf.sqrt(tf.reduce_mean(tf.square(tf.sub(target, output))))
    return cost


def tf_cnn_model(num_conv_layers, num_filters, filter_size,
                 num_fc_layers, num_units_in_fc,  
                 img_height, img_width, img_channels, dim_output,
                 img_resize=None, strides=None, padding='SAME',
                 stddev=0.05, max_pooling=None, use_relu=True):
    
    
    x = tf.placeholder(tf.float32, shape=[None, img_height*img_width*img_channels], name='x')
    x_image = tf.reshape(x, [-1, img_height, img_width, img_channels])

    if img_resize is not None:
    #downsample the images
        x_image = tf.image.resize_images(x_image, size=[img_resize['width'], img_resize['height']], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)

    y = tf.placeholder(dtype=tf.float32, shape=[None, dim_output], name="y")

    layer_conv = x_image
    weights_of_cnn_layers = []
    num_inp_channels = img_channels

    for k in range(num_conv_layers):
        layer_conv, weights_conv = create_conv_layer(data_in=layer_conv,
                                                     num_inp_channels=num_inp_channels,
                                                     filter_size=filter_size[k],
                                                     num_filters=num_filters[k],
                                                     strides=strides,
                                                     padding=padding,
                                                     stddev=stddev,
                                                     max_pooling=max_pooling)

        num_inp_channels = num_filters[k]
        weights_of_cnn_layers.append(weights_conv)

    layer_shape = layer_conv.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer_conv, [-1, num_features])

    layer_fc = layer_flat
    num_inputs = num_features

    for k in range(num_fc_layers):
        num_outputs = num_units_in_fc[k]
        layer_fc = create_fc_layer(data_in=layer_fc, 
                                   num_inputs=num_inputs, 
                                   num_outputs=num_outputs, 
                                   stddev=stddev, 
                                   use_relu=use_relu)

        num_inputs = num_outputs

    loss = get_loss_cnn(layer_fc, y)
    train_op = get_train(loss)

    output_ops = {'output' : layer_fc, 'loss': loss, 'weights_of_cnn_layers': weights_of_cnn_layers, 'train': train_op, 'x': x, 'y': y}

    return output_ops





