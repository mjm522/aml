import tensorflow as tf
import numpy as np

import os, sys



# Adding slim path
# tf_slim_path = os.environ['TF_SLIM_PATH']
# tf_slim_path = os.path.abspath(os.path.join(tf_slim_path, 'models/slim', ''))
# sys.path.append(tf_slim_path)

from nets import inception
from preprocessing import inception_preprocessing
slim = tf.contrib.slim
image_size = inception.inception_v1.default_image_size

def init_weights(shape, name=None):
    return tf.get_variable(name, initializer=tf.random_normal(shape, stddev=0.01))


def init_bias(shape, name=None):
    return tf.get_variable(name, initializer=tf.zeros(shape, dtype='float'))


def batched_matrix_vector_multiply(vector, matrix):
    """ computes x^T A in mini-batches. """
    vector_batch_as_matricies = tf.expand_dims(vector, [1])
    mult_result = tf.batch_matmul(vector_batch_as_matricies, matrix)
    squeezed_result = tf.squeeze(mult_result, [1])
    return squeezed_result


def euclidean_loss_layer(a, b, precision, batch_size):
    """ Math:  out = (action - mlp_out)'*precision*(action-mlp_out)
                    = (u-uhat)'*A*(u-uhat)"""
    scale_factor = tf.constant(2*batch_size, dtype='float')
    uP = batched_matrix_vector_multiply(a-b, precision)
    uPu = tf.reduce_sum(uP*(a-b))  # this last dot product is then summed, so we just the sum all at once.
    return uPu/scale_factor

def euclidean_loss(a, b, batch_size):
    scale_factor = tf.constant(2*batch_size, dtype='float')
    squared_diff = tf.square(a-b)

    cost = tf.reduce_sum(squared_diff)

    return cost/scale_factor

def get_input_layer(dim_input, dim_output):
    net_input = tf.placeholder("float", [None, dim_input], name='nn_input')
    # Ground truth position
    position = tf.placeholder('float', [None, dim_output], name='position')

    return net_input, position

def get_mlp_layers(mlp_input, number_layers, dimension_hidden):
    """compute MLP with specified number of layers.
        math: sigma(Wx + b)
        for each layer, where sigma is by default relu"""
    cur_top = mlp_input
    weights = []
    biases = []
    for layer_step in range(0, number_layers):
        in_shape = cur_top.get_shape().dims[1].value
        cur_weight = init_weights([in_shape, dimension_hidden[layer_step]], name='w_' + str(layer_step))
        cur_bias = init_bias([dimension_hidden[layer_step]], name='b_' + str(layer_step))
        weights.append(cur_weight)
        biases.append(cur_bias)
        if layer_step != number_layers-1:  # final layer has no RELU
            cur_top = tf.nn.relu(tf.matmul(cur_top, cur_weight) + cur_bias)
        else:
            cur_top = tf.matmul(cur_top, cur_weight) + cur_bias

    return cur_top, weights, biases


def get_loss_layer(mlp_out, position, batch_size):
    """The loss layer used for the MLP network is obtained through this class."""
    return euclidean_loss(a=position, b=mlp_out, batch_size=batch_size)


def pose_estimation_network(dim_input=27, dim_output=9, batch_size=25, network_config=None):
    """
    An example a network in tf that has both state and image inputs.
    Args:
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
        network_config: dictionary of network structure parameters
    Returns:
        A tfMap object that stores inputs, outputs, and scalar loss.
    """
    n_layers = 3
    layer_size = 20
    dim_hidden = (n_layers - 1)*[layer_size]
    dim_hidden.append(dim_output)
    filter_size = 5

    net_input, position = get_input_layer(dim_input, dim_output)


    # image goes through 3 convnet layers
    num_filters = network_config['num_filters']

    im_height = network_config['image_height']
    im_width = network_config['image_width']
    num_channels = network_config['image_channels']
    image_input = tf.reshape(net_input, [-1, num_channels, im_width, im_height])
    image_input = tf.transpose(image_input, perm=[0,3,2,1])

    # Store layers weight & bias
    with tf.variable_scope('conv_params'):
        weights = {
            'wc1': init_weights([7, 7, num_channels, num_filters[0]], name='wc1'), # 5x5 conv, 1 input, 32 outputs
            'wc2': init_weights([filter_size, filter_size, num_filters[0], num_filters[1]], name='wc2'), # 5x5 conv, 32 inputs, 64 outputs
            'wc3': init_weights([filter_size, filter_size, num_filters[1], num_filters[2]], name='wc3'), # 5x5 conv, 32 inputs, 64 outputs
        }

        biases = {
            'bc1': init_bias([num_filters[0]], name='bc1'),
            'bc2': init_bias([num_filters[1]], name='bc2'),
            'bc3': init_bias([num_filters[2]], name='bc3'),
        }

    conv_layer_0 = conv2d(img=image_input, w=weights['wc1'], b=biases['bc1'], strides=[1,2,2,1])
    conv_layer_1 = conv2d(img=conv_layer_0, w=weights['wc2'], b=biases['bc2'])
    conv_layer_2 = conv2d(img=conv_layer_1, w=weights['wc3'], b=biases['bc3'])


    _, num_rows, num_cols, num_fp = conv_layer_2.get_shape()
    num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]
    x_map = np.empty([num_rows, num_cols], np.float32)
    y_map = np.empty([num_rows, num_cols], np.float32)

    for i in range(num_rows):
        for j in range(num_cols):
            x_map[i, j] = (i - num_rows / 2.0) / num_rows
            y_map[i, j] = (j - num_cols / 2.0) / num_cols

    x_map = tf.convert_to_tensor(x_map)
    y_map = tf.convert_to_tensor(y_map)

    x_map = tf.reshape(x_map, [num_rows * num_cols])
    y_map = tf.reshape(y_map, [num_rows * num_cols])

    # rearrange features to be [batch_size, num_fp, num_rows, num_cols]
    features = tf.reshape(tf.transpose(conv_layer_2, [0,3,1,2]),
                          [-1, num_rows*num_cols])
    softmax = tf.nn.softmax(features)

    fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
    fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)

    fp = tf.reshape(tf.concat(axis=1, values=[fp_x, fp_y]), [-1, num_fp*2])

    fc_input = fp

    # _, num_rows, num_cols, num_fp = conv_layer_2.get_shape()

    # conv_layer_2_size = (im_height*im_width*num_filters[2])/4

    # fc_input = tf.reshape(conv_layer_2, [-1,conv_layer_2_size])

    fc_output, _, _ = get_mlp_layers(fc_input, n_layers, dim_hidden)

    loss = euclidean_loss(a=position, b=fc_output, batch_size=batch_size)

    return {'input': net_input, 'position': position, 'fc_out': fc_output, 'loss': loss, 'features' : fp}

def conv2d(img, w, b, strides=[1, 1, 1, 1]):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=strides, padding='SAME'), b))


def get_xavier_weights(filter_shape, poolsize=(2, 2)):
    fan_in = np.prod(filter_shape[1:])
    fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
               np.prod(poolsize))

    low = -4*np.sqrt(6.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation
    high = 4*np.sqrt(6.0/(fan_in + fan_out))
    return tf.Variable(tf.random_uniform(filter_shape, minval=low, maxval=high, dtype=tf.float32))

def train_adam_step(loss_op):
    return tf.train.AdamOptimizer(1e-4).minimize(loss_op)


def load_inception_model(network_config, net_input = None):

    with slim.arg_scope(inception.inception_v1_arg_scope()):
        
        if net_input is None:
            net_input = tf.placeholder("float", [None, image_size, image_size, 3], name='nn_input')
        
        logits, end_points = inception.inception_v1(net_input, num_classes=1001, is_training=True)
        
        init_fn = slim.assign_from_checkpoint_fn(
        os.path.join('./checkpoints', 'inception_v1.ckpt'),
        slim.get_model_variables('InceptionV1'))
        
        return logits, end_points, init_fn, net_input
    

def get_fp_layer(conv_layer_in):
    _, num_rows, num_cols, num_fp = conv_layer_in.get_shape()
    num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]
    x_map = np.empty([num_rows, num_cols], np.float32)
    y_map = np.empty([num_rows, num_cols], np.float32)

    for i in range(num_rows):
        for j in range(num_cols):
            x_map[i, j] = (i - num_rows / 2.0) / num_rows
            y_map[i, j] = (j - num_cols / 2.0) / num_cols

    x_map = tf.convert_to_tensor(x_map)
    y_map = tf.convert_to_tensor(y_map)

    x_map = tf.reshape(x_map, [num_rows * num_cols])
    y_map = tf.reshape(y_map, [num_rows * num_cols])

    # rearrange features to be [batch_size, num_fp, num_rows, num_cols]
    features = tf.reshape(tf.transpose(conv_layer_in, [0,3,1,2]),
                          [-1, num_rows*num_cols])
    softmax = tf.nn.softmax(features)

    fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
    fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)

    fp = tf.reshape(tf.concat(axis=1, values=[fp_x, fp_y]), [-1, num_fp*2])
    
    return fp


def get_conv_weights(key_w = 'wc2',key_b = 'bc2', filter_size = 5, input_channels = 64, output_channels = 64):
    with tf.variable_scope('conv_params'):
        weights = {
            key_w: init_weights([filter_size, filter_size, input_channels, output_channels], name=key_w), # 5x5 conv, 32 inputs, 64 outputs        
        }

        biases = {
            key_b: init_bias([output_channels], name=key_b),
        }
        
    
    
        return weights, biases



def get_input_layer_fp(dim_input, dim_output, dim_pose_output = 9):
    """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to commpute loss."""
    net_input = tf.placeholder("float", [None, dim_input], name='nn_input')
    action = tf.placeholder('float', [None, dim_output], name='action')
    precision = tf.placeholder('float', [None, dim_output, dim_output], name='precision')

    pose = tf.placeholder('float', [None, dim_pose_output], name='pose')

    return net_input, action, precision, pose


def multi_modal_network_fp(dim_input=27, dim_output=7, batch_size=25, network_config=None):
    """
    An example a network in tf that has both state and image inputs, with the feature
    point architecture (spatial softmax + expectation).
    Args:
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
        network_config: dictionary of network structure parameters
    Returns:
        A tfMap object that stores inputs, outputs, and scalar loss.
    """
    n_layers = 3
    layer_size = 20
    dim_hidden = (n_layers - 1)*[layer_size]
    dim_hidden.append(dim_output)

    dim_pose_output = 3
    dim_pose_hidden = (n_layers - 1)*[layer_size]
    dim_pose_hidden.append(dim_pose_output)

    pool_size = 2
    filter_size = 5

    # List of indices for state (vector) data and image (tensor) data in observation.
    x_idx, img_idx, i = [], [], 0
    for sensor in network_config['obs_include']:
        dim = network_config['sensor_dims'][sensor]
        if sensor in network_config['obs_image_data']:
            img_idx = img_idx + list(range(i, i+dim))
        else:
            x_idx = x_idx + list(range(i, i+dim))
        i += dim

    nn_input, action, precision, pose = get_input_layer_fp(dim_input, dim_output, dim_pose_output)


    state_input = nn_input[:, 0:x_idx[-1]+1]
    image_input = nn_input[:, x_idx[-1]+1:img_idx[-1]+1]

    # image goes through 3 convnet layers
    num_filters = network_config['num_filters']

    im_height = network_config['image_height']
    im_width = network_config['image_width']
    num_channels = network_config['image_channels']
    image_input = tf.reshape(image_input, [-1, num_channels, im_width, im_height])
    image_input = tf.transpose(image_input, perm=[0,3,2,1])

    # we pool twice, each time reducing the image size by a factor of 2.
    conv_out_size = int(im_width/(2.0*pool_size)*im_height/(2.0*pool_size)*num_filters[1])
    first_dense_size = conv_out_size + len(x_idx)

    # Store layers weight & bias
    with tf.variable_scope('conv_params'):
        weights = {
            # 'wc1': init_weights([filter_size, filter_size, num_channels, num_filters[0]], name='wc1'), # 5x5 conv, 1 input, 32 outputs
            'wc2': init_weights([filter_size, filter_size, 64, num_filters[1]], name='wc2'), # 5x5 conv, 32 inputs, 64 outputs
            'wc3': init_weights([filter_size, filter_size, num_filters[1], num_filters[2]], name='wc3'), # 5x5 conv, 32 inputs, 64 outputs
        }

        biases = {
            # 'bc1': init_bias([num_filters[0]], name='bc1'),
            'bc2': init_bias([num_filters[1]], name='bc2'),
            'bc3': init_bias([num_filters[2]], name='bc3'),
        }


    image_input_inception = inception_preprocessing.preprocess_image(image_input[0], im_height, im_width, is_training=True)

    logits, end_points, init_fn, image_input_inception = load_inception_model(network_config,image_input)



    conv_layer_0 = end_points['Conv2d_1a_7x7']

    # conv_layer_0 = conv2d(img=image_input, w=weights['wc1'], b=biases['bc1'], strides=[1,2,2,1])
    conv_layer_1 = conv2d(img=conv_layer_0, w=weights['wc2'], b=biases['bc2'])
    conv_layer_2 = conv2d(img=conv_layer_1, w=weights['wc3'], b=biases['bc3'])

    
    fp = get_fp_layer(conv_layer_2)

    fc_action_input = tf.concat(axis=1, values=[fp, state_input])
    fc_pose_input = fp

    fc_action_output, weights_FC_ACTION, biases_FC_ACTION = get_mlp_layers(fc_action_input, n_layers, dim_hidden)
    fc_action_vars = weights_FC_ACTION + biases_FC_ACTION


    with tf.variable_scope('pose_net'):
        fc_pose_output, weights_FC_POSE, biases_FC_POSE = get_mlp_layers(fc_pose_input, n_layers, dim_pose_hidden)
        fc_pose_vars = weights_FC_POSE + biases_FC_POSE

    loss_action = euclidean_loss_layer(a=action, b=fc_action_output, precision=precision, batch_size=batch_size)

    loss_pose = euclidean_loss(a=pose, b=fc_pose_output, batch_size=batch_size)

    last_conv_vars = fc_action_input

    net_dict = {'nn_input': nn_input, 
                'pose': pose, 
                'action': action,
                'precision': precision,
                'fc_pose_out': fc_pose_output,
                'fc_action_out': fc_action_output, 
                'loss_pose': loss_pose, 
                'loss_action': loss_action,
                'last_conv_vars': last_conv_vars,
                'fc_action_vars': fc_action_vars,
                'features' : fp, 
                'init_fn': init_fn}
    

    return net_dict