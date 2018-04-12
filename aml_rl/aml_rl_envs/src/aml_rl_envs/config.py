import copy
from os.path import abspath

urdf_root_path = '/'.join(abspath(__file__).split('/')[:-1] ) + '/models/'


AML_RL_ENV_CONFIG = {
	'renders':True,
	'ctrl_type':'pos',
	'max_steps':5000,
	'time_step':1./240.,
	'urdf_root_path':urdf_root_path,
	'cam_dist':1.3,
	'cam_yaw':-180.,
	'cam_pitch':-40,
	'large_val_observation':100,
	'render_height':720,
	'render_width':960,

}

AML_RL_ROBOT_CONFIG = copy.deepcopy(AML_RL_ENV_CONFIG)

AML_RL_ROBOT_CONFIG['max_force'] = 200
AML_RL_ROBOT_CONFIG['max_velocity'] = 0.35




