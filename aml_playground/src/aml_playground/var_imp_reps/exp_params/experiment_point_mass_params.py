import os
import copy
import numpy as np
from aml_rl_envs.point_mass.config import POINT_MASS_ENV_CONFIG, POINT_MASS_CONFIG

gpreps_params = {
    'w_dim':4,
    'initial_params': 0. * np.ones(4),
    'entropy_bound':2.,
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
gpreps_params_1['w_dim'] = 6
gpreps_params_1['initial_params'] = 0. * np.ones(6)
gpreps_params_1['context_dim'] = 6
gpreps_params_1['context_feature_dim'] = 6
gpreps_params_1['x_dim']= 3
gpreps_params_1['sp_x_dim'] = 3
gpreps_params_1['sp_y_dim'] = 3
gpreps_params_1['w_bounds'] = np.array([[ 0.,    0.,    0.,    -np.sqrt(0),    -np.sqrt(0.),    -np.sqrt(0)],
                                        [ 1.,    1.,    1.,    np.sqrt(1.),    np.sqrt(1.),    np.sqrt(1.)]])

point_mass = copy.deepcopy(POINT_MASS_CONFIG)
point_mass_env =  copy.deepcopy(POINT_MASS_ENV_CONFIG)

#robot config modification
point_mass['enable_force_torque_sensors'] = True
point_mass['ctrl_type'] = 'pos'


#environment param modification
point_mass_env['robot_config'] = point_mass
point_mass_env['spring_stiffness'] = 0.
point_mass_env['f_dot_weight'] = 0.
point_mass_env['work_weight'] = 0.
point_mass_env['goal_weight'] = 0.5
point_mass_env['finishing_weight'] = 2
point_mass_env['f_des_weight'] = 0.
point_mass_env['u_weight'] = 0.
point_mass_env['num_traj_points'] = 100
point_mass_env['reward_gamma'] = 1.
point_mass_env['ramp_traj_flag'] = False

experiment_1 = {
    'experiment_name':'exp_1',
    'max_itr':100,
    'gpreps_params':gpreps_params_1,
    'env_params':point_mass_env,
    'param_file_name':os.environ['AML_DATA'] + '/aml_playground/imp_worlds/creps_data_point_mass_smooth_off_1_with_sigmoid.pkl',
}

experiment_2 =  copy.deepcopy(experiment_1)
experiment_2['env_params']['u_weight'] = 0.001
experiment_2['param_file_name']=os.environ['AML_DATA'] + '/aml_playground/imp_worlds/creps_data_point_mass_smooth_off_2.pkl'

exp_params = experiment_2