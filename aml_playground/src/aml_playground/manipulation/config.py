import os
import copy
import numpy as np
from aml_rl_envs.hand.config import HAND_CONFIG
from aml_rl_envs.hand.config import HAND_OBJ_CONFIG

HAND_OBJ_CONFIG['ctrl_type'] = 'torque'

network_params = {
    'load_saved_model':False,
    'model_path':os.environ['AML_DATA']+'/aml_playground/manipulation/immitation/keras_models/simple_nn.h5',
    'epochs':500,
    'batch_size':20,
    'save_model':True,
    'state_dim':2,
    'cmd_dim':1,
    'out_dim':3,
}


IMMITATION_CONFIG = {
	'network_params':network_params,
}
