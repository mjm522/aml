import os
import numpy as np

#kpx kpy kpz
sg ={
'start':np.array([10., 10., 10.]),
'goal':np.array([10.,10., 10.]),
}

state_constraints={
'min':[10., 10., 10.],
'max':[70., 70., 70.],
}


experiment_1 = {
	'timesteps':100,
	'no_rollouts':100,
	'h':10,
	'gain':1e-1,
	'max_iter':61,
	'state_constraints':state_constraints,
	'start_goal':sg,
	'init_traj':None,
	'cost_fn':None,
}

ee_config = experiment_1