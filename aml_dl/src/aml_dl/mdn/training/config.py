
from os.path import dirname, abspath
from aml_robot.box2d.config import config

check_point_path = dirname(dirname(abspath(__file__))) + '/training/tf_check_points/'

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