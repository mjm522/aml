point_env_config = {
    'dt':0.01,
    'state_lim':[-1.,1.],
    'action_lim':[-1.,1.]
}

adam_params = {
    'type': 'adam',
    'params': {'learning_rate' : 0.0001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-08, 'use_locking': False}
}

reps_config = {
    'feature_dim':8, #env._state_dim * 2 + 4
    'discount':0.9, 
    'gae_lambda':0.9,
    'positive_adv':False,
    'center_adv':False,
    'epsilon':0.5, 
    'L2_reg_dual':0., 
    'L2_reg_loss':0., 
    'max_opt_itr':50,
    'tf_opt_params':adam_params,
    
}

policy_config = {
    'feature_dim':reps_config['feature_dim'],
    'output_dim':2,
    'hidden_sizes':(32, 32),
    'learn_std':True,
    'init_std':1.0,
    'adaptive_std':False,
    'std_share_network':False,
    'std_hidden_sizes':(32, 32),
    'min_std':1e-6,
    'std_hidden_nonlinearity':'tanh',
    'hidden_nonlinearity':'tanh',
    'output_nonlinearity':None,
    'mean_network':None,
    'std_network':None,
}



demo_config = {
    'env_config':point_env_config,
    'algo_config':reps_config,
    'policy_config':policy_config,
}