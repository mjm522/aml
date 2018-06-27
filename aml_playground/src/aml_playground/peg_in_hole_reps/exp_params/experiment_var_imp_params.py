import os
import copy
import numpy as np
from experiment_params import gpreps_params

gpreps_params_1 = copy.deepcopy(gpreps_params)
gpreps_params_1['w_dim'] = 6
gpreps_params_1['initial_params'] = 0. * np.ones(6)
gpreps_params_1['context_dim'] = 3
gpreps_params_1['context_feature_dim'] = 10
gpreps_params_1['x_dim']= 3
gpreps_params_1['w_bounds'] = np.array([[-0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
                                        [ 0.5, 0.5,  0.5, 0.5, 0.5,  0.5]])

experiment_1 = {
    
    'gpreps_params':gpreps_params_1,
}


exp_params = experiment_1