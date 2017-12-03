import os
import copy
import numpy as np

from aml_planners.push_planner.forward_models.gp_model import GPModel
from aml_planners.push_planner.forward_models.ensemble_model import EnsambleModel
from aml_planners.push_planner.forward_models.simple_nn_model import SimpleNNModel
from aml_planners.push_planner.push_worlds.config import push_world_config as config

EXP_NAME = 'exp_ensemble'

MPPI_DATA_DIR     = os.environ['AML_DATA'] + '/aml_planners/push_planner/single_push_planner/'
check_point_dir   = MPPI_DATA_DIR + 'tf_check_points/exp_ensemble/'
summary_dir       = MPPI_DATA_DIR + 'exp_ensemble/'

if not os.path.exists(check_point_dir):
    os.makedirs(check_point_dir)

if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)


random_seed = 42


adam_params = {

    'type': 'adam',
    'params': {'learning_rate' : 0.001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-08, 'use_locking': False}
}

#two fingers
config['num_fins'] = 2

mppi_params ={

    'start':None,
    'goal':None,
    'obstacle':None,

    'traj_path': 'paths/follow_path_some_shape.pkl', # should be none for single goal point

    'obstacle_radius':None,

    'dynamics_type': 'learnt_dyn', #'learnt_dyn', 'box2d'

    'max_iterations':2000,
    'dt': 0.1,

    'resample_proportion':0.,
    'task_complete_thresh':0.9,

    'goal_ori_coeff': 0.,

    'use_cutoff_heuristic': False, # Used for cost_imp only at the moment
    'cutoff_heuristic_thr': 200,

    'use_mc_unc_estimation': False,

    'N':3,
    'h':2.,
    'rho': 1.5,#10.7,#0.2
    'K':10,
    'cmd_dim': 2,
    'state_dim': 6,
    'random_seed': random_seed,
    'init_policy': 'use_random',
    'cmd_filter': np.array([0,0,1,1]),
    'R': np.eye(2)*0.01,
    'Q': np.eye(6)*0.01,
    'nu': 100,
    'goal_cost_coeff': 5.0,#0.85,#0.05,#0.35,
    'control_cost_coeff': 0.001*0,
    'obstacle_cost_coeff':0.65*0,
    'vel_state_cost_coeff': 0.003*0,
    'uncertain_cost_coeff': 5.0,
    'final_cost_coeff':2.,
    }


""" 

Experiment 1: Box2d real robot : 

"""

mppi_params_1 = copy.deepcopy(mppi_params)

mppi_params_1['start'] = np.array([0., 0., 0.026, 0., 0., 0.])

mppi_params_1['goal'] = np.array([0.69396412,  0.67552775,  1.63927054, 0., 0., 0.]) #goal on right side 1

mppi_params_1['obstacle'] = None 

mppi_params_1['max_iterations'] = 2000
mppi_params_1['dt'] = 0.05#0.009
mppi_params_1['N'] = 2 #5
mppi_params_1['K'] = 150 #150
mppi_params_1['h'] = 1.0 #1.
mppi_params_1['rho'] = 1.0
mppi_params_1['cmd_dim'] = 2
mppi_params_1['R'] = np.eye(mppi_params_1['cmd_dim'])*0.5

mppi_params_1['dynamics_type'] = 'box2d' #'box2d' 'learnt_dyn'0
mppi_params_1['goal_cost_coeff'] =  1.5 #1.5 # 10.5
mppi_params_1['goal_ori_coeff'] = 0.01 # 0.01
mppi_params_1['control_cost_coeff'] = 0.0
mppi_params_1['obstacle_cost_coeff'] = 3.5
mppi_params_1['vel_state_cost_coeff'] = 0.0
mppi_params_1['uncertain_cost_coeff'] = 0.0#4.0

mppi_params_1['use_cutoff_heuristic'] = False
mppi_params_1['cutoff_heuristic_thr'] = 180
mppi_params_1['task_complete_thresh'] = 0.07

config['dt'] = mppi_params_1['dt']
config['goal'] = mppi_params_1['goal'].copy()[:3]

adam_params = {

    'type': 'adam',
    'params': {'learning_rate' : 0.001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-08, 'use_locking': False}
}

experiment_1 = {

    'push_world_config': config,
    'forward_model': GPModel,#EnsambleModel, GPModel # Used by learnt dynamics only
    'data_path': MPPI_DATA_DIR + 'baxter_push_data_combined.csv',

    'is_quasi_static': True,

    'dynamics_type': mppi_params_1['dynamics_type'],
    'cost_type': 'cost_2push_imp',

    'box2D_viewer': False,

    'transform_angle_baxter_2_box2d':-0.5*np.pi,

    'show_obstacle':False,

    'show_goal':True,

    'dt': mppi_params_1['dt'],

    'random_seed': random_seed,

    'mppi_params':mppi_params_1,

    'exp_image_folder' : MPPI_DATA_DIR + 'experiment_images/exp_8_push_plus_x_image',

    'savefig': False,

    'network_params':{
            'train_iterations': 5000,
            'n_ensembles': 10,
            'dim_input': 6, 
            'dim_output': 3,
            'n_hidden': [20,20,20,20,20], #[20,20,20,20,20]
            'k_mixtures': 1,
            'write_summary': False,
            'load_saved_model': False,
            'model_dir': check_point_dir,
            'model_name':'ensemble_model_u_shape.ckpt',
            'optimiser': adam_params,
            'summary_dir':summary_dir,
            'device': '/cpu:0',
            'adv_epsilon': 0.001,
    }
    
}

## Select experiment here
experiment_config = experiment_1