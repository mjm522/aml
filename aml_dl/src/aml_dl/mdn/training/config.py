import os
from aml_robot.box2d.config import config
from aml_io.io_tools import get_aml_package_path

check_point_path   = get_aml_package_path('aml_dl') + '/src/aml_dl/mdn/training/tf_check_points/'
training_data_path = ''
try:
    training_data_path = os.environ['AML_DATA'] + '/aml_dl/baxter_push_data/'
except:
    print "AML_DATA environment variable is not set."

if not os.path.exists(training_data_path):
    print "Training data folder does not exist..."
    #raise ValueError


NUM_FP         = 10 #this is not the right value?
IMAGE_WIDTH    = config['image_width']
IMAGE_HEIGHT   = config['image_height']
IMAGE_CHANNELS = 3

network_params_inv = {
    'num_filters': [5, 5, NUM_FP],
    'dim_input': 4, 
    'dim_output': 1,
    'n_hidden': 1,
    'k_mixtures': 2,
    'batch_size': 25,
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'image_size': IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
    'load_saved_model': True,
    'model_path': check_point_path + 'push_model.ckpt',
    'device': '/cpu:0',
}

train_file_indices = range(1,5)
test_file_indices  = range(41,49)

network_params_fwd = {
    'num_filters': [5, 5, NUM_FP],
    'dim_input': 9, 
    'dim_output': 7,
    'output_order':['qt_w','qt_x','qt_y','qt_z','x','y','z'],
    'n_hidden_layers': 2, #including the input layer excluding the output layer
    'units_in_hidden_layers':[9,9],#pass none is all layers have equal to input
    'batch_size': 25,
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'image_size': IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
    'load_saved_model': True,
    'model_path': check_point_path+'push_model_fwd.ckpt',
    'training_data_path':training_data_path,
    'train_file_indices':train_file_indices,
    'test_file_indices':test_file_indices,
    'device': '/cpu:0',
}

num_conv_layers = 3
filter_sizes_each_layer = [5, 5, 5]
num_filters_per_layer = [16,32,16]

if len(filter_sizes_each_layer) != num_conv_layers:
    print "Number of filter sizes in each layer has to specified"
    raise ValueError

if len(num_filters_per_layer) != num_conv_layers:
    print "Number of filters in each layer has to specified"
    raise ValueError

network_params_cnn = {
    'num_conv_layers':num_conv_layers,
    'num_fc_layers':2,
    'num_units_in_fc':[128,7],
    'filter_sizes':filter_sizes_each_layer,
    'num_filters':num_filters_per_layer,
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'dim_output':7,
    'max_pooling':{'x':2, 'y':2},
    'strides':[1,1,1,1],
    'padding':'SAME',
    'use_relu':True,
    'stddev':0.05,
    'load_saved_model': True,
    'model_path': check_point_path+'push_model_fwd.ckpt',
    'training_data_path':training_data_path,
    'train_file_indices':train_file_indices,
    'test_file_indices':test_file_indices,
    'device':'/cpu:0',

}