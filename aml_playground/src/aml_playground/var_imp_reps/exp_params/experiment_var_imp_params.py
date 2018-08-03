import os
import copy
import numpy as np
from experiment_params import gpreps_params
from aml_rl_envs.sawyer.config import SAWYER_CONFIG, SAWYER_ENV_CONFIG

gpreps_params_1 = copy.deepcopy(gpreps_params)
gpreps_params_1['w_dim'] = 6
gpreps_params_1['initial_params'] = 0. * np.ones(6)
gpreps_params_1['context_dim'] = 9
gpreps_params_1['context_feature_dim'] = 9
gpreps_params_1['x_dim']= 3
gpreps_params_1['sp_x_dim'] = 3
gpreps_params_1['sp_y_dim'] = 3
gpreps_params_1['w_bounds'] = np.array([[-0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
                                        [ 3  ,    3,    3,    3,    3,    3]])

sawyer_1 = copy.deepcopy(SAWYER_CONFIG)
sawyer_env_1 =  copy.deepcopy(SAWYER_ENV_CONFIG)

#robot config modification
sawyer_1['enable_force_torque_sensors'] = True
sawyer_1['ctrl_type'] = 'pos'


#environment param modification
sawyer_env_1['robot_config'] = sawyer_1
sawyer_env_1['spring_stiffness'] = 2
sawyer_env_1['f_dot_weight'] = 1.5
sawyer_env_1['work_weight'] = 0.5
sawyer_env_1['goal_weight'] = 0.5
sawyer_env_1['f_des_weight'] = 1.
sawyer_env_1['u_weight'] = 0.01


experiment_1 = {
    'experiment_name':'exp_1',
    'max_itr':400,
    'gpreps_params':gpreps_params_1,
    'env_params':sawyer_env_1,
    'param_file_name':os.environ['AML_DATA'] + '/aml_playground/imp_worlds/reps_data_local_Jul_26_2018_15_22_57.pkl',
}


exp_params = experiment_1