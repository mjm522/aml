import os
import numpy as np

gpreps_params = {
	'w_dim':2,
	'initial_params': 4.0 * np.ones(2),
	'entropy_bound':2.0,
	'context_dim':2,
	'context_feature_dim':6,
	'x_dim':4, #input param dimension of GP
	'policy_variance':0.03,
	'random_state':np.random.RandomState(0),
	'num_policy_updates':25,
	'num_old_datasets':1,
	'min_eta':1e-8, 
	'num_data_to_collect':20, 
	'num_fake_data':30,
	'num_samples_per_update':30,
	'num_samples_fwd_data':50,
}


experiment_1 = {
	
	'gpreps_params':gpreps_params,
}

exp_params = experiment_1