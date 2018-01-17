import numpy as np


sg ={
'start':np.array([0.75, 6.75]),
'goal':np.array([6.5, 0.75]),
'obstacle':np.array([3.5,3.5]),
'r_obs':1.0,
}

state_constraints={
'min':[0., 0.],
'max':[7., 7.],
}


config_1 = {
	'timesteps':100,
	'no_rollouts':100,
	'h':10,
	'gain':1e-1,
	'max_iter':61,
	'state_constraints':state_constraints,
	'start_goal':sg,
	'init_traj':None,
	'cost_fn':None,
	'smooth_traj':True,
}

ex_config = config_1