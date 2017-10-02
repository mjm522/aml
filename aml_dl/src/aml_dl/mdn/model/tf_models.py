import numpy as np
import tensorflow as tf
from aml_dl.utilities.tf_optimisers import optimiser_op
from tf_mdn_model import MixtureDensityNetwork

def weight_variable(shape, stddev=0.05):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable(shape, stddev=0.1):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(stddev, shape=shape)
    return tf.Variable(initial)

def cnn_layer(input_tensor, layer_name, num_inp_channels, filter_size,  num_filters, activate, 
              max_pooling=None, strides=None, padding='SAME', stddev=0.05):
    
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

def get_quadratic_loss(output, target):
    diff = output-target
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.matmul(diff, tf.transpose(diff)), keep_dims=True)), keep_dims=True)

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
    weight_layer_name = 'fc_weight'
    bias_layer_name   = 'fc_bias'
    sum_layer_name    = 'fc_Wx_plus_b'
    
    if layer_type=='cnn':
        weight_layer_name = 'cnn_weight'
        bias_layer_name   = 'cnn_bias'
        sum_layer_name    = 'cnn_Wx_plus_b'
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
            with tf.name_scope(weight_layer_name):
                tf_sumry_wrtr.add_variable_summaries(weights)
            with tf.name_scope(bias_layer_name):
                tf_sumry_wrtr.add_variable_summaries(biases)
            with tf.name_scope(sum_layer_name):
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

def tf_model(dim_input, dim_output, loss_type, cnn_params, fc_params, optimiser_params, tf_sumry_wrtr):
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

    with tf.name_scope('cost_fwd'):
        if loss_type == 'quadratic':
            cost = get_quadratic_loss(fc_layer_output, target)

    with tf.name_scope('train'):
        train_step = optimiser_op(cost, optimiser_params)
 
    if tf_sumry_wrtr is not None:
        tf_sumry_wrtr.add_scalar(name_scope='cost', data=cost)
        # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        tf_sumry_wrtr.write_summary()

    output_ops = {'output' : fc_layer_output, 'cost': cost, 'train_step': train_step, 'x': x, 'image_input':image_input, 'y': target}
    return output_ops

def tf_siamese_model(loss_type, cnn_params, fc_params, optimiser_params, mdn_params, cost_weights, tf_sumry_wrtr, ):
    fc_params  = configure_params(fc_params)
    cnn_params = configure_params(cnn_params)

    image_len = cnn_params['image_width']*cnn_params['image_height']*cnn_params['image_channels']
    image_input_t   = tf.placeholder(dtype=tf.float32, shape=[None, image_len],   name='x_t-input')
    image_input_t_1 = tf.placeholder(dtype=tf.float32, shape=[None, image_len],   name='x_t_1-input')
    
    with tf.name_scope('input_reshape'):
        image_shaped_input_t   = tf.transpose(tf.reshape(image_input_t, [-1, cnn_params['image_channels'], cnn_params['image_width'], cnn_params['image_height']]), perm=[0,3,2,1])
        image_shaped_input_t_1 = tf.transpose(tf.reshape(image_input_t_1, [-1, cnn_params['image_channels'], cnn_params['image_width'], cnn_params['image_height']]), perm=[0,3,2,1])

        img_resize_params = cnn_params['img_resize']

        if  img_resize_params is not None:
            image_len = img_resize_params['width']*img_resize_params['height']*cnn_params['image_channels']
            with tf.name_scope('input_resize'):
                image_shaped_input_t   = tf.image.resize_images(image_shaped_input_t,   size=[img_resize_params['height'], img_resize_params['width']], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
                image_shaped_input_t_1 = tf.image.resize_images(image_shaped_input_t_1, size=[img_resize_params['height'], img_resize_params['width']], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
        
        if tf_sumry_wrtr is not None:
            tf_sumry_wrtr.add_image(name_scope='input_resize_t',   image=image_shaped_input_t)
            tf_sumry_wrtr.add_image(name_scope='input_resize_t_1', image=image_shaped_input_t_1)

    def conv_relu(input, variable_name, weight_name, bias_name, kernel_shape, bias_shape, stride, activate, padding='SAME', max_pooling=None, pre_activate_scope=None, activate_scope=None):
        # Create variable named "weights".
        with tf.variable_scope(variable_name):
            weights = tf.get_variable(weight_name, kernel_shape, initializer=tf.random_normal_initializer())
            # Create variable named "biases".
            biases = tf.get_variable(bias_name, bias_shape, initializer=tf.constant_initializer(0.0))
        
        with tf.variable_scope(pre_activate_scope):
            conv = tf.nn.conv2d(input, weights, strides=stride, padding=padding)

        pre_activations = conv + biases

        if max_pooling is not None:
            pool_ksize   = [1, max_pooling['x'], max_pooling['y'], 1]
            pool_strides = [1, max_pooling['x'], max_pooling['y'], 1]
            layer = tf.nn.max_pool(value=result, ksize=pool_ksize, strides=pool_strides, padding=padding) 

        with tf.variable_scope(activate_scope):
            activations  = activate(conv + biases)

        return weights, biases, pre_activations, activations

    def image_filter(input_images, params):
        no_channels = params['image_channels']
        for k in range(params['num_layers']):
            with tf.variable_scope(params['layer_names'][k]):
                kernel_shape = [params['filter_sizes'][k], params['filter_sizes'][k], no_channels, params['num_filters'][k]]
                no_channels = params['num_filters'][k]
                bias_shape = params['num_filters'][k]  #for clarity
                weights, biases, pre_activations, activations = conv_relu(input=input_images, 
                                                           variable_name=params['variable_names'][k], 
                                                           weight_name=params['weight_names'][k], 
                                                           bias_name=params['bias_names'][k], 
                                                           kernel_shape=kernel_shape, 
                                                           bias_shape=bias_shape,
                                                           stride=params['strides'][k],
                                                           activate=params['layer_activations'][k],
                                                           padding=params['padding'][k],
                                                           max_pooling=params['max_pooling'][k],
                                                           pre_activate_scope=params['pre_activate_scope'][k], 
                                                           activate_scope=params['activate_scope'][k])
                if tf_sumry_wrtr is not None:
                    with tf.name_scope('cnn_weight'):
                        tf_sumry_wrtr.add_variable_summaries(weights)
                    with tf.name_scope('cnn_bias'):
                        tf_sumry_wrtr.add_variable_summaries(biases)
                    with tf.name_scope('cnn_Wx_plus_b'):
                        tf_sumry_wrtr.add_histogram(name_scope='pre_activations', data=pre_activations)
                    tf_sumry_wrtr.add_histogram(name_scope='activations', data=activations)

                if params['use_dropout'][k]:
                    with tf.name_scope('dropout'):
                        keep_prob = tf.placeholder(tf.float32)
                        if tf_sumry_wrtr is not None:
                            tf_sumry_wrtr.add_scalar(name_scope='dropout_keep_probability', data=keep_prob)
                            dropped = tf.nn.dropout(result, keep_prob)
                            result = dropped
    
                input_images = activations

        return activations

    with tf.variable_scope("image_filters") as scope:
        cnn_layer_output_t = image_filter(image_shaped_input_t, cnn_params)
        scope.reuse_variables()
        cnn_layer_output_t_1 = image_filter(image_shaped_input_t_1, cnn_params)
        if tf_sumry_wrtr is not None:
            tf_sumry_wrtr.add_variable_summaries(cnn_layer_output_t)
            tf_sumry_wrtr.add_variable_summaries(cnn_layer_output_t_1)


    # layer_shape_t      = cnn_layer_output_t.get_shape() 
    # num_features_t     = layer_shape_t[1:4].num_elements()
    # layer_flat_t       = tf.reshape(cnn_layer_output_t, [-1, num_features_t])

    # layer_shape_t_1      = cnn_layer_output_t_1.get_shape()
    # num_features_t_1     = layer_shape_t_1[1:4].num_elements()
    # layer_flat_t_1       = tf.reshape(cnn_layer_output_t_1, [-1, num_features_t_1])

    def get_feature_points(cnn_layer_out):

        _, num_rows, num_cols, num_fp = cnn_layer_out.get_shape()
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
        features = tf.reshape(tf.transpose(cnn_layer_out, [0,3,1,2]),
                              [-1, num_rows*num_cols])
        softmax = tf.nn.softmax(features)
        fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
        fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)
        fp = tf.reshape(tf.concat(axis=1, values=[fp_x, fp_y]), [-1, num_fp*2])
        return fp

    with tf.variable_scope('feature_points_t'):
        feature_point_t = get_feature_points(cnn_layer_output_t)

    with tf.variable_scope('feature_points_t_1'):
        feature_point_t_1 = get_feature_points(cnn_layer_output_t_1)

    if tf_sumry_wrtr is not None:
        tf_sumry_wrtr.add_variable_summaries(feature_point_t)
        tf_sumry_wrtr.add_variable_summaries(feature_point_t_1)

    ##########################################Pluging in MDN ####################################
    _, num_fp = feature_point_t.get_shape()
    num_fp = int(num_fp)

    mdn_input_op = tf.concat(axis=1,values=[feature_point_t, feature_point_t_1])
    _, dim_input = mdn_input_op.get_shape()
    dim = int(dim_input) 
    mdn_params['dim_input'] = dim

    mdn = MixtureDensityNetwork(network_params=mdn_params, tf_sumry_wrtr = tf_sumry_wrtr) 
    mdn._init_model(input_op=mdn_input_op) # Construct model graph

    ############################################FORWARD MODEL######################################

    fwd_model_inp = tf.concat(axis=1,values=[feature_point_t, mdn._ops['y']])

    fc_layer_output  = create_nn_layers(fwd_model_inp, fc_params, tf_sumry_wrtr, layer_type='fc')

    # with tf.name_scope('input_y'):
    #     target = tf.placeholder(dtype=tf.float32, shape=[None, dim_output],  name='y-input')

    with tf.name_scope('cost_fwd'):
        if loss_type == 'quadratic':
            print fc_layer_output
            print feature_point_t_1
            print "**********************************************************************"
            cost_fwd = get_quadratic_loss(fc_layer_output, feature_point_t_1)

    ###########################COMBINED COST FUNCTION##############################################
    with tf.name_scope('total_cost'):
        cost_total  =  tf.reduce_mean(cost_weights['fwd']*cost_fwd + cost_weights['inv']*mdn._ops['loss'])

    with tf.name_scope('train'):
        train_step = optimiser_op(cost_total, optimiser_params)

    ######################################WRAP UP##################################################
    
    if tf_sumry_wrtr is not None:
        tf_sumry_wrtr.add_scalar(name_scope='cost_fwd',   data=tf.reduce_mean(cost_fwd))
        # tf_sumry_wrtr.add_scalar(name_scope='cost_inv',   data=tf.reduce_mean(mdn._ops['loss']))
        tf_sumry_wrtr.add_scalar(name_scope='cost_total', data=cost_total)
        # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        tf_sumry_wrtr.write_summary()

    output_ops = {'output' : fc_layer_output, 
                  'cost': cost_total, 
                  'train_step': train_step, 
                  'image_input_t': image_input_t, 
                  'image_input_t_1': image_input_t_1,
                  'fwd_cost': cost_fwd,
                  'mdn_y': mdn._ops['y'], # mdn target is a push action
                  'mdn_loss' : mdn._ops['loss'],
                  'mdn_mu' : mdn._ops['mu'],
                  'mdn_pi' : mdn._ops['pi'],
                  'mdn_sigma' : mdn._ops['sigma'],
                  'fp_xt': feature_point_t,
                  'fp_xt1': feature_point_t_1}
    return output_ops



