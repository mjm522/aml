import copy
from aml_rl_envs.config import AML_RL_ENV_CONFIG
from aml_rl_envs.config import AML_RL_ROBOT_CONFIG

POINT_MASS_ENV_CONFIG = copy.deepcopy(AML_RL_ENV_CONFIG)
POINT_MASS_CONFIG = copy.deepcopy(AML_RL_ROBOT_CONFIG)

POINT_MASS_CONFIG['is_discrete'] = False
POINT_MASS_ENV_CONFIG['action_repeat'] = 1

POINT_MASS_CONFIG['ctrl_type'] = 'torque'
POINT_MASS_ENV_CONFIG['ctrl_type'] = 'torque'
POINT_MASS_ENV_CONFIG['spring_stiffness'] = 0.
POINT_MASS_ENV_CONFIG['ramp_traj_flag'] = False
POINT_MASS_ENV_CONFIG['num_traj_points'] = 100
POINT_MASS_ENV_CONFIG['num_traj_points'] = 100
POINT_MASS_ENV_CONFIG['finishing_weight'] = 0.
POINT_MASS_ENV_CONFIG['u_weight'] = 0.
POINT_MASS_ENV_CONFIG['f_des_weight'] = 0.
POINT_MASS_ENV_CONFIG['goal_weight']= 0.5
POINT_MASS_ENV_CONFIG['reward_gamma'] = 0.99
