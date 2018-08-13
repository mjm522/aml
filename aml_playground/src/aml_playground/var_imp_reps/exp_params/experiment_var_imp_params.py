import os
import copy
import numpy as np
from aml_rl_envs.sawyer.config import SAWYER_CONFIG, SAWYER_ENV_CONFIG

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


gpreps_params_1 = copy.deepcopy(gpreps_params)
gpreps_params_1['w_dim'] = 7
gpreps_params_1['initial_params'] = 0. * np.ones(6)
gpreps_params_1['context_dim'] = 9
gpreps_params_1['context_feature_dim'] = 9
gpreps_params_1['x_dim']= 3
gpreps_params_1['sp_x_dim'] = 3
gpreps_params_1['sp_y_dim'] = 3
gpreps_params_1['w_bounds'] = np.array([[ 0,    0,    0,    0,    0,    0,  0],
                                        [ 5,    5,    5,    5,    5,    5,  5]])

sawyer_1 = copy.deepcopy(SAWYER_CONFIG)
sawyer_env_1 =  copy.deepcopy(SAWYER_ENV_CONFIG)

#robot config modification
sawyer_1['enable_force_torque_sensors'] = True
sawyer_1['ctrl_type'] = 'pos'


#environment param modification
sawyer_env_1['robot_config'] = sawyer_1
sawyer_env_1['spring_stiffness'] = 0
sawyer_env_1['f_dot_weight'] = 1.5
sawyer_env_1['work_weight'] = 0.5
sawyer_env_1['goal_weight'] = 0.9
sawyer_env_1['f_des_weight'] = 0.
sawyer_env_1['u_weight'] = 0.0
sawyer_env_1['num_traj_points'] = 100
sawyer_env_1['reward_gamma'] = 0.9
sawyer_env_1['ramp_traj_flag'] = False

experiment_1 = {
    'experiment_name':'exp_1',
    'max_itr':100,
    'gpreps_params':gpreps_params_1,
    'env_params':sawyer_env_1,
    'param_file_name':os.environ['AML_DATA'] + '/aml_playground/imp_worlds/creps_data_only_x_jnt_space_stiffness.pkl',
}


exp_params = experiment_1