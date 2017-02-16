
import os.path

from aml_robot.box2d.config import config
from aml_io.io_tools import get_aml_package_path

check_point_path   = get_aml_package_path('aml_dl') + '/src/aml_dl/mdn/training/tf_check_points/'
training_data_path = get_aml_package_path('aml_data_collec_utils') + '/data/simulation/'

if not os.path.exists(training_data_path):
    print "Training data folder does not exist..."
    raise ValueError


NUM_FP         = 10 #this is not the right value?
IMAGE_WIDTH    = config['image_width']
IMAGE_HEIGHT   = config['image_height']
IMAGE_CHANNELS = 3

network_params = {
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

network_params_fwd = {
    'num_filters': [5, 5, NUM_FP],
    'dim_input': 10, 
    'dim_output': 7,
    'n_hidden_layers': 2, #including the input layer excluding the output layer
    'units_in_hidden_layers':[9,7],#pass none is all layers have equal to input
    'batch_size': 25,
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'image_size': IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
    'load_saved_model': True,
    'model_path': check_point_path + 'sim_push_model_fwd.ckpt',
    'training_data_path':training_data_path,
    'train_data_file_names':['sim_push_data_01.pkl'],
    'device': '/cpu:0',
}