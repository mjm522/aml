import numpy as np
import tensorflow as tf

def weight_variable(shape, stddev=0.05):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable(shape, stddev=0.1):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(stddev, shape=shape)
    return tf.Variable(initial)


def cnn_layer(input_tensor, layer_name, num_inp_channels, filter_size,  num_filters, activate, max_pooling=None, strides=None, padding='SAME', stddev=0.05):
    
    with tf.name_scope(layer_name):
        # Shape of the filter-weights for the convolution.
        shape    = [filter_size, filter_size, num_inp_channels, num_filters]
        # Create new weights aka. filters with the given shape.
        weights  = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
        # Create new biases, one for each filter.
        biases   = tf.Variable(tf.constant(stddev, shape=[num_filters]))

        if strides is None:
            strides = [1,1,1,1]
        # Create the TensorFlow operation for convolution.
        layer    = tf.nn.conv2d(input=input_tensor, filter=weights, strides=strides, padding=padding)
        # Add the biases to the results of the convolution.
        layer   += biases

        # Use pooling to down-sample the image resolution
        if max_pooling is not None:
          pool_ksize   = [1, max_pooling['x'], max_pooling['y'], 1]
          pool_strides = [1, max_pooling['x'], max_pooling['y'], 1]
          layer = tf.nn.max_pool(value=layer, ksize=pool_ksize, strides=pool_strides, padding=padding)

        activations = activate(layer, name='activation')

    return weights, biases, layer, activations

def nn_layer(input_tensor, layer_name, output_dim, activate, max_pooling=None):

    layer_shape = input_tensor.get_shape()
    input_dim   = layer_shape[1:4].num_elements()

    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        weights  = weight_variable([input_dim, output_dim])
        biases   = bias_variable([output_dim])
        layer = tf.matmul(input_tensor, weights)

        layer += biases

        if max_pooling is not None:
            pool_ksize   = [1, max_pooling['x'], max_pooling['y'], 1]
            pool_strides = [1, max_pooling['x'], max_pooling['y'], 1]
            layer = tf.nn.max_pool(value=layer, ksize=pool_ksize, strides=pool_strides, padding=padding) 
        
        activations = activate(layer, name='activation')
        return weights, biases, layer, activations


def get_loss_cnn(output, target):
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target)
    # cost = tf.reduce_mean(cross_entropy)
    with tf.name_scope('cost'):
        cost =  tf.square(tf.sub(target, output))
    with tf.name_scope('total'):
        total = tf.reduce_mean(cost)

    return cost, total

def get_quadratic_loss(output, target):
    with tf.name_scope('cost'):
        cost =  tf.square(tf.sub(target, output))
    with tf.name_scope('total'):
        total = tf.sqrt(tf.reduce_mean(cost))

    return cost, total

def configure_params(params):
    #usually strides and max_pooling happens for cnn layers and not fc layers, to 
    #take care of that
    num_layers = params['num_layers']

    layer_activations = []
    for act in params['activation']:
        if act == 'relu':
            layer_activations.append(tf.nn.relu)
        elif act =='idty':
            layer_activations.append(tf.identity)
        else:
            print "Unknown action given"
            raise ValueError

    params['layer_activations'] = layer_activations

    return params

def create_nn_layers(inp, params, tf_sumry_wrtr, layer_type):
    params = configure_params(params)
    num_layers        = params['num_layers']
    layer_names       = params['layer_names']
    layer_outputs     = params['layer_outputs']
    layer_activations = params['layer_activations']
    max_pooling       = params['max_pooling']
    use_dropout       = params['use_dropout']
    
    if layer_type=='cnn':
        num_inp_channels  = params['image_channels']
        filter_sizes      = params['filter_sizes']
        num_filters       = params['num_filters']
        strides           = params['strides']
        padding           = params['padding']
   
    layer_input = inp

    for k in range(num_layers):

        if layer_type=='fc':

            weights, biases, preactivate, activations = nn_layer(input_tensor=layer_input,
                                                                 layer_name=layer_names[k], 
                                                                 output_dim=layer_outputs[k], 
                                                                 activate=layer_activations[k], 
                                                                 max_pooling=max_pooling[k])
        elif layer_type=='cnn':

            weights, biases, preactivate, activations = cnn_layer(input_tensor=layer_input,
                                                                 layer_name=layer_names[k],
                                                                 num_inp_channels=num_inp_channels, 
                                                                 filter_size=filter_sizes[k],
                                                                 num_filters=num_filters[k],
                                                                 activate=layer_activations[k], 
                                                                 max_pooling=max_pooling[k], 
                                                                 strides=strides[k], 
                                                                 padding=padding[k])

            num_inp_channels = num_filters[k]

        if tf_sumry_wrtr is not None:
            with tf.name_scope('weights'):
                tf_sumry_wrtr.add_variable_summaries(weights)
            with tf.name_scope('biases'):
                tf_sumry_wrtr.add_variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                tf_sumry_wrtr.add_histogram(name_scope='pre_activations', data=preactivate)
            tf_sumry_wrtr.add_histogram(name_scope='activations', data=activations)

        if use_dropout[k]:
            with tf.name_scope('dropout'):
                keep_prob = tf.placeholder(tf.float32)
                if tf_sumry_wrtr is not None:
                    tf_sumry_wrtr.add_scalar(name_scope='dropout_keep_probability', data=keep_prob)
                dropped = tf.nn.dropout(activations, keep_prob)

            layer_input = dropped
        else:
            layer_input = activations

    return layer_input


def tf_model(dim_input, dim_output, loss_type, learning_rate, cnn_params, fc_params, tf_sumry_wrtr):
    # Create a multilayer model.
    image_input = None
    
    if cnn_params is not None:

        image_len = cnn_params['image_width']*cnn_params['image_height']*cnn_params['image_channels']
        image_input  = tf.placeholder(dtype=tf.float32, shape=[None, image_len],   name='x-input')
        #in case we are not resizing
        x = image_input
        
        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(image_input, [-1, cnn_params['image_width'], cnn_params['image_height'], cnn_params['image_channels']])
            img_resize_params = cnn_params['img_resize']

            if  img_resize_params is not None:
                image_len = img_resize_params['width']*img_resize_params['height']*cnn_params['image_channels']
                with tf.name_scope('input_resize'):
                    image_shaped_input = tf.image.resize_images(image_shaped_input, size=[img_resize_params['width'], img_resize_params['height']], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
                    x = tf.reshape(image_shaped_input, [-1, image_len], name='x-input')
            
            if tf_sumry_wrtr is not None:
                tf_sumry_wrtr.add_image(name_scope='input_resize', image=image_shaped_input)

        cnn_layer_output = create_nn_layers(image_shaped_input, cnn_params, tf_sumry_wrtr, layer_type='cnn')
        layer_shape = cnn_layer_output.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(cnn_layer_output, [-1, num_features])
    
    else:
         # Input placeholders
        with tf.name_scope('input'):
            layer_flat = tf.placeholder(dtype=tf.float32, shape=[None, dim_input],  name='x-input')
            x = layer_flat
    
    with tf.name_scope('input'):
        target = tf.placeholder(dtype=tf.float32, shape=[None, dim_output],  name='y-input')

    fc_layer_output  = create_nn_layers(layer_flat, fc_params, tf_sumry_wrtr, layer_type='fc')

    if loss_type == 'normal':
        cost, total = get_loss_cnn(fc_layer_output, target)
    elif loss_type == 'quadratic':
        cost, total = get_quadratic_loss(fc_layer_output, target)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(fc_layer_output, 1), tf.argmax(target, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    if tf_sumry_wrtr is not None:
        tf_sumry_wrtr.add_scalar(name_scope='cost', data=total)
        tf_sumry_wrtr.add_scalar(name_scope='accuracy', data=accuracy)
        # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        tf_sumry_wrtr.write_summary()

    output_ops = {'output' : fc_layer_output, 'cost': total, 'accuracy':accuracy, 'train_step': train_step, 'x': x, 'image_input':image_input, 'y': target}
    return output_ops



