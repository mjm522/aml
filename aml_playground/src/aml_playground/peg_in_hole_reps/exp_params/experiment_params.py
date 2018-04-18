import os
import copy
import numpy as np

gpreps_params = {
	'w_dim':4,
	'initial_params': 0. * np.ones(4),
	'entropy_bound':2.0,
	'context_dim':3,
	'context_feature_dim':10,
	'x_dim':10, #input param dimension of GP
	'policy_variance':0.03,
	'random_state':np.random.RandomState(0),
	'num_policy_updates':25,
	'num_old_datasets':1,
	'min_eta':1e-8, 
	'num_data_to_collect':20, 
	'num_fake_data':30,
	'num_samples_per_update':30,
	'num_samples_fwd_data':50,
	'w_bounds': np.array([[-0.015, -0.013, 0., 0.],[ 0.015,  0.013, 0., 0.05]])
}


experiment_1 = {
	
	'gpreps_params':gpreps_params,
}

#context learning for changing orientation of the peg -5 to 5 degree
gpreps_params_2 = copy.deepcopy(gpreps_params)
gpreps_params_2['w_dim'] = 5
gpreps_params_2['initial_params'] = 0. * np.ones(5)
gpreps_params_2['context_dim'] = 1
gpreps_params_2['context_feature_dim'] = 3
gpreps_params_2['x_dim']= 3
gpreps_params_2['w_bounds'] = np.array([[-0.015, -0.013, 0., 0., -0.1],
	                                    [ 0.015,  0.013, 0., 0.05, 0.1]])

experiment_2 = {
	
	'gpreps_params':gpreps_params_2,
}

#context learning for changing orientation of the peg -12 to 12 degree
gpreps_params_3 = copy.deepcopy(gpreps_params)
gpreps_params_3['w_dim'] = 5
gpreps_params_3['initial_params'] = 0. * np.ones(5)
gpreps_params_3['context_dim'] = 1
gpreps_params_3['context_feature_dim'] = 3
gpreps_params_3['x_dim']= 3
gpreps_params_3['w_bounds'] = np.array([[-0.015, -0.013, 0., 0., -0.2],
	                                    [ 0.015,  0.013, 0., 0.05, 0.2]])

experiment_3 = {
	
	'gpreps_params':gpreps_params_3,
}

#context learning for changing orientation of the peg -5 to 5 degree
#additionally, Large Kp is also penalized
gpreps_params_4 = copy.deepcopy(gpreps_params)
gpreps_params_4['w_dim'] = 8
gpreps_params_4['initial_params'] = 0. * np.ones(8)
gpreps_params_4['context_dim'] = 1
gpreps_params_4['context_feature_dim'] = 3
gpreps_params_4['x_dim']= 3
gpreps_params_4['w_bounds'] = np.array([[-0.015, -0.013, 0., 0., -0.1, -0.5, -0.5, -0.5],
	                                    [ 0.015,  0.013, 0., 0.05, 0.1, 0.5, 0.5,  0.5]])

experiment_4 = {
	
	'gpreps_params':gpreps_params_4,
}



exp_params = experiment_2