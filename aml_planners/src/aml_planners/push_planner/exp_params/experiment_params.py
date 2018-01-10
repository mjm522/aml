import os
import copy
import numpy as np

from aml_planners.push_planner.forward_models.gp_model import GPModel
from aml_planners.push_planner.forward_models.ensemble_model import EnsambleModel
try:
    from aml_planners.push_planner.forward_models.simple_nn_model import SimpleNNModel
except:
    SimpleNNModel = EnsambleModel
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



sg_1 = {
    'start':np.array([2.5, 6.75, 0., 0., 0., 0.]),
    'goal':np.array([6.5, 3.5, 0., 0., 0., 0.]),
    'obstacle':np.array([3.5,3.5]),
}

sg_2 = {
    'start':np.array([1.0, 6.0, 0., 0., 0., 0.]),
    'goal':np.array([6.0, 1.0, 0., 0., 0., 0.]),
    'obstacle':np.array([3.5,3.5]),
}

sg_4 = {
    'start':np.array([1.0, 6.0, 0., 0., 0., 0.]),
    'goal':np.array([5.0, 1.0, 0., 0., 0., 0.]),
    'obstacle':np.array([3.5,3.5]),
}


mppi_params ={

    'start':sg_1['start'],
    'goal':sg_1['goal'],
    'obstacle':None,

    'traj_path': 'paths/follow_path_some_shape.pkl', # should be none for single goal point

    'obstacle_radius':1.,

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
    'update_rho':False,
    'update_RhoN':25,
    'rhoMin':0.01,
    'rhoMax':1.2,
    }


box2dViewer_config = {

    'image_width': 640,
    'image_height': 480,
    'no_samples':20,
    'fps': 15,
    'dt': 0.05  ,#0.0167,
    'steps_per_frame': 15,
    'window_caption': 'BoxWorld',
    'pixels_per_meter': 200,
    'push_mag': 10.0,
    'pre_push_offset':0.1,
}

# Default MPPI configs
# dt = 0.05, K = 10, N = 3, rho = 2.0, h = 5.0, cmd_dim = 2, state_dim = 6, seed = 42

""" 

Experiment 1: TRAJECTORY FOLLOWING (Trajectory pre computed so as to dodge uncertainty) 


"""
mppi_params_1 = copy.deepcopy(mppi_params)

mppi_params_1['start'] = sg_1['start']
mppi_params_1['goal'] = sg_1['goal']
mppi_params_1['obstacle'] = sg_1['obstacle']

mppi_params_1['traj_path'] = 'paths/follow_path_some_shape_modified.pkl'
mppi_params_1['dynamics_type'] = 'learnt_dyn'
mppi_params_1['max_iterations'] = 2000
mppi_params_1['dt'] = 0.1
mppi_params_1['N'] = 2
mppi_params_1['h'] = 2
mppi_params_1['rho'] = 1.5



mppi_params_1['goal_cost_coeff'] = 5.0
mppi_params_1['control_cost_coeff'] = 0.0
mppi_params_1['obstacle_cost_coeff'] = 0.0
mppi_params_1['vel_state_cost_coeff'] = 0.0
mppi_params_1['uncertain_cost_coeff'] = 0.0
mppi_params_1['uncertain_cost_coeff'] = 2.0


experiment_1 = {

    'push_world_config': config,
    'forward_model': EnsambleModel, # Used by learnt dynamics only
    'data_path': MPPI_DATA_DIR + 'push_data_good.csv',

    'box2D_viewer': False,

    'is_quasi_static': True,

    'dynamics_type': mppi_params_1['dynamics_type'],

    'dt': mppi_params_1['dt'],
    
    'cost_type': 'cost_traj_following',

    'random_seed': random_seed,

    'mppi_params':mppi_params_1,

    'network_params': {
            'train_iterations': 500,
            'n_ensembles': 20,
            'dim_input': 10, 
            'dim_output': 6,
            'n_hidden': [20, 20, 20, 20],
            'k_mixtures': 1,
            'write_summary': False,
            'load_saved_model': False,
            'model_dir': check_point_dir,
            'model_name':'ensemble_model_u_shape.ckpt',
            'optimiser': adam_params,
            'summary_dir':summary_dir,
            'device': '/cpu:0',
            'adv_epsilon': 0.005,
    }
    
}


'''
experiment_1 with new params
'''

mppi_params_1 = copy.deepcopy(mppi_params)

mppi_params_1['start'] = sg_1['start']
mppi_params_1['goal'] = sg_1['goal']
mppi_params_1['obstacle'] = sg_1['obstacle']

mppi_params_1['traj_path'] = 'paths/follow_path_some_shape_modified.pkl'
mppi_params_1['dynamics_type'] = 'learnt_dyn'
mppi_params_1['max_iterations'] = 10000
mppi_params_1['dt'] = 0.1
mppi_params_1['N'] = 2
mppi_params_1['K'] = 100
mppi_params_1['h'] = 2
mppi_params_1['rho'] = 2.5 



mppi_params_1['goal_cost_coeff'] = 9.0
mppi_params_1['control_cost_coeff'] = 0.0
mppi_params_1['obstacle_cost_coeff'] = 0.0
mppi_params_1['vel_state_cost_coeff'] = 0.0
mppi_params_1['uncertain_cost_coeff'] = 0.0
mppi_params_1['uncertain_cost_coeff'] = 2.0

experiment_1_1 = {

    'push_world_config': config,
    'forward_model': EnsambleModel, # Used by learnt dynamics only
    'data_path': MPPI_DATA_DIR + 'push_data_good.csv',

    'box2D_viewer': False,

    'is_quasi_static': True,

    'dynamics_type': mppi_params_1['dynamics_type'],

    'dt': mppi_params_1['dt'],
    
    'cost_type': 'cost_traj_following',

    'random_seed': random_seed,

    'mppi_params':mppi_params_1,

    'network_params': {
            'train_iterations': 500,
            'n_ensembles': 20,
            'dim_input': 10, 
            'dim_output': 6,
            'n_hidden': [20, 20, 20, 20],
            'k_mixtures': 1,
            'write_summary': False,
            'load_saved_model': False,
            'model_dir': check_point_dir,
            'model_name':'ensemble_model_u_shape.ckpt',
            'optimiser': adam_params,
            'summary_dir':summary_dir,
            'device': '/cpu:0',
            'adv_epsilon': 0.005,
    }
    
}


""" 

Experiment 2: Heuristic Uncertainty Avoidance 

"""

mppi_params_2 = copy.deepcopy(mppi_params)

mppi_params_2['start'] = sg_2['start']
mppi_params_2['goal'] = sg_2['goal']
mppi_params_2['obstacle'] = sg_2['obstacle']

# mppi_params_2['traj_path'] = 'paths/follow_path_some_shape.pkl'
mppi_params_2['dynamics_type'] = 'learnt_dyn'
mppi_params_2['max_iterations'] = 2000
mppi_params_2['dt'] = 0.05
mppi_params_2['N'] = 2
mppi_params_2['K'] = 10
mppi_params_2['R'] = np.eye(2)*0.5
mppi_params_2['h'] = 2.0
mppi_params_2['rho'] = 2.0


mppi_params_2['goal_cost_coeff'] = 0.35
mppi_params_2['control_cost_coeff'] = 0.0
mppi_params_2['obstacle_cost_coeff'] = 0.0
mppi_params_2['vel_state_cost_coeff'] = 0.0
mppi_params_2['uncertain_cost_coeff'] = 65.0#4.0

mppi_params_2['use_cutoff_heuristic'] = True
mppi_params_2['cutoff_heuristic_thr'] = 180


experiment_2 = {

    'push_world_config': config,
    'forward_model': EnsambleModel, # Used by learnt dynamics only
    'data_path': MPPI_DATA_DIR + 'push_data_good.csv',

    'is_quasi_static': True,

    'dynamics_type': 'learnt_dyn', #'learnt_dyn', 'box2d'
    'cost_type': 'cost_imp',

    'box2D_viewer': False,

    'dt': mppi_params_2['dt'],

    'random_seed': random_seed,

    'mppi_params':mppi_params_2,

    'savefig': False,

    'network_params': {
            'train_iterations': 500,
            'n_ensembles': 10,
            'dim_input': 10, 
            'dim_output': 6,
            'n_hidden': [15],
            'k_mixtures': 1,
            'write_summary': False,
            'load_saved_model': False,
            'model_dir': check_point_dir,
            'model_name':'ensemble_model_u_shape.ckpt',
            'optimiser': adam_params,
            'summary_dir':summary_dir,
            'device': '/cpu:0',
            'adv_epsilon': 0.005,
    }

    

    
}


""" 

Experiment 3: Scaled by Uncertainty Cost 

"""

mppi_params_3 = copy.deepcopy(mppi_params)

mppi_params_3['start'] = sg_2['start']
mppi_params_3['goal'] = sg_2['goal']
mppi_params_3['obstacle'] = sg_2['obstacle']

# mppi_params_2['traj_path'] = 'paths/follow_path_some_shape.pkl'
mppi_params_3['dynamics_type'] = 'learnt_dyn'
mppi_params_3['max_iterations'] = 2000
mppi_params_3['dt'] = 0.05
mppi_params_3['N'] = 2
mppi_params_3['K'] = 10
mppi_params_3['R'] = np.eye(2)*0.5
mppi_params_3['h'] = 2.0
mppi_params_3['rho'] = 1.5


mppi_params_3['goal_cost_coeff'] = 0.1
mppi_params_3['control_cost_coeff'] = 0.0
mppi_params_3['obstacle_cost_coeff'] = 0.0
mppi_params_3['vel_state_cost_coeff'] = 0.0
mppi_params_3['uncertain_cost_coeff'] = 0.25

mppi_params_3['use_cutoff_heuristic'] = False
mppi_params_3['cutoff_heuristic_thr'] = 200


experiment_3 = {

    'push_world_config': config,
    'forward_model': EnsambleModel, # Used by learnt dynamics only
    'data_path': MPPI_DATA_DIR + 'push_data_bottom_left.csv',

    'is_quasi_static': True,

    'dynamics_type': 'box2d', #'learnt_dyn', 'box2d'
    'cost_type': 'cost_scl_unc',

    'box2D_viewer': False,

    'dt': mppi_params_3['dt'],

    'random_seed': random_seed,

    'mppi_params':mppi_params_3,


    'network_params': {
            'train_iterations': 500,
            'n_ensembles': 10,
            'dim_input': 10, 
            'dim_output': 6,
            'n_hidden': [25],
            'k_mixtures': 1,
            'write_summary': False,
            'load_saved_model': False,
            'model_dir': check_point_dir,
            'model_name':'ensemble_model_u_shape.ckpt',
            'optimiser': adam_params,
            'summary_dir':summary_dir,
            'device': '/cpu:0',
            'adv_epsilon': 0.005,
    }

    

    
}


""" 

Experiment 4: Uncertainty tradeoff

"""

mppi_params_4 = copy.deepcopy(mppi_params)

mppi_params_4['start'] = sg_4['start']
mppi_params_4['goal'] = sg_4['goal']
mppi_params_4['obstacle'] = sg_4['obstacle']

# mppi_params_2['traj_path'] = 'paths/follow_path_some_shape.pkl'
mppi_params_4['dynamics_type'] = 'learnt_dyn'
mppi_params_4['max_iterations'] = 2000
mppi_params_4['dt'] = 0.05
mppi_params_4['N'] = 2
mppi_params_4['K'] = 10
mppi_params_4['R'] = np.eye(2)*0.5
mppi_params_4['h'] = 5.0
mppi_params_4['rho'] = 1.5


mppi_params_4['goal_cost_coeff'] = 0.35
mppi_params_4['control_cost_coeff'] = 0.0
mppi_params_4['obstacle_cost_coeff'] = 0.0
mppi_params_4['vel_state_cost_coeff'] = 0.0
mppi_params_4['uncertain_cost_coeff'] = 6.5

mppi_params_4['use_cutoff_heuristic'] = False
mppi_params_4['cutoff_heuristic_thr'] = 200

mppi_params_4['use_mc_unc_estimation'] = False

mppi_params_4['init_policy'] = 'use_random' # use_guess, use_random


world_config_4 = copy.deepcopy(config)
world_config_4['force_mag'] = 15


# 20, n_seq_pushes = 1, noise = 0.1
experiment_4 = {

    'push_world_config': world_config_4,
    'forward_model': EnsambleModel, # Used by learnt dynamics only
    'data_path': MPPI_DATA_DIR + 'push_data_bottom_left.csv',

    'is_quasi_static': True,

    'dynamics_type': 'learnt_dyn', #'learnt_dyn', 'box2d'
    'cost_type': 'cost_imp',

    'box2D_viewer': False,

    'dt': mppi_params_4['dt'],

    'random_seed': random_seed,

    'mppi_params':mppi_params_4,


    'network_params': {
            'train_iterations': 500,
            'n_ensembles': 10,
            'dim_input': 10, 
            'dim_output': 6,
            'n_hidden': [20],
            'k_mixtures': 1,
            'write_summary': False,
            'load_saved_model': False,
            'model_dir': check_point_dir,
            'model_name':'ensemble_model_u_shape.ckpt',
            'optimiser': adam_params,
            'summary_dir':summary_dir,
            'device': '/cpu:0',
            'adv_epsilon': 0.005,
    }

    

    
}

""" 

Experiment 5: Surface push parametarization: Working params with MPPI 

"""

mppi_params_5 = copy.deepcopy(mppi_params)

mppi_params_5['start'] = sg_2['start']
mppi_params_5['goal'] = sg_2['goal']
mppi_params_5['obstacle'] = sg_2['obstacle']

# mppi_params_5['traj_path'] = 'paths/follow_path_some_shape.pkl'
mppi_params_5['dynamics_type'] = 'box2d'
mppi_params_5['max_iterations'] = 2000
mppi_params_5['dt'] = 0.05
mppi_params_5['N'] = 10
mppi_params_5['K'] = 150
mppi_params_5['h'] = 1.0
mppi_params_5['rho'] = 1.0
mppi_params_5['cmd_dim'] = 1
mppi_params_5['R'] = np.eye(mppi_params_5['cmd_dim'])*0.5


mppi_params_5['goal_cost_coeff'] = 1.5
mppi_params_5['control_cost_coeff'] = 0.0
mppi_params_5['obstacle_cost_coeff'] = 0.0
mppi_params_5['vel_state_cost_coeff'] = 0.0
mppi_params_5['uncertain_cost_coeff'] = .0#4.0

mppi_params_5['use_cutoff_heuristic'] = True
mppi_params_5['cutoff_heuristic_thr'] = 180


experiment_5 = {

    'push_world_config': config,
    'forward_model': EnsambleModel, # Used by learnt dynamics only
    'data_path': MPPI_DATA_DIR + 'push_data_good.csv',

    'is_quasi_static': True,

    'dynamics_type': 'box2d', #'learnt_dyn', 'box2d'
    'cost_type': 'cost_imp',

    'box2D_viewer': False,

    'dt': mppi_params_5['dt'],

    'random_seed': random_seed,

    'mppi_params':mppi_params_5,

    'savefig': False,

    'network_params': {
            'train_iterations': 500,
            'n_ensembles': 10,
            'dim_input': 10, 
            'dim_output': 6,
            'n_hidden': [15],
            'k_mixtures': 1,
            'write_summary': False,
            'load_saved_model': False,
            'model_dir': check_point_dir,
            'model_name':'ensemble_model_u_shape.ckpt',
            'optimiser': adam_params,
            'summary_dir':summary_dir,
            'device': '/cpu:0',
            'adv_epsilon': 0.005,
    }

    

    
}

""" 

Experiment 6: Surface push parametarization, simple NN

"""

mppi_params_6 = copy.deepcopy(mppi_params)

#0.564768850803,0.652302980423,0.0,0.80308477767,1.81704521179,0.621471226215,-0.486279308796
#0.529103398323,0.436444014311,0.0,0.3146949954,-0.725897789001,0.435279250145,-0.0190378446132
#0.534361839294,0.323695987463,0.0,0.0013805292809,0.714055001736,1.49824011326,-3.15912151337
#0.282166570425,0.395610362291,0.0,0.71113346217,0.359045237303,-0.845775485039,1.21135723591

#baxter data
#0.732159655755,0.276926887194,0.234879401945,0.0694952029321,0.738926599904,0.280368261557,0.264172482399
#0.717686359481,0.354216314659,0.0468301270816,0.396402005453,0.705682277723,0.419571833116,0.336554239022
#0.658223314259,0.358930527107,-0.192333118685,0.654332799806,0.662278483406,0.358735667891,-0.25399025971
#0.723033090322,0.353576904386,0.050489315064,0.814498266455,0.805259560549,0.381742693044,0.709408880844
# mppi_params_6['start'] = np.array([0.717686359481,0.354216314659, 0., 0., 0., 0.])
# mppi_params_6['goal'] = np.array([0.705682277723,0.419571833116, 0., 0., 0., 0.])


# mppi_params_5['traj_path'] = 'paths/follow_path_some_shape.pkl'
mppi_params_6['dynamics_type'] = 'box2d'
mppi_params_6['max_iterations'] = 2000
mppi_params_6['dt'] = 0.05
mppi_params_6['N'] = 5
mppi_params_6['K'] = 150
mppi_params_6['h'] = 1.0
mppi_params_6['rho'] = 1.0
mppi_params_6['cmd_dim'] = 1
mppi_params_6['R'] = np.eye(mppi_params_6['cmd_dim'])*0.5


mppi_params_6['goal_cost_coeff'] = 1.5
mppi_params_6['control_cost_coeff'] = 0.0
mppi_params_6['obstacle_cost_coeff'] = 0.0
mppi_params_6['vel_state_cost_coeff'] = 0.0
mppi_params_6['uncertain_cost_coeff'] = .0#4.0

mppi_params_6['use_cutoff_heuristic'] = True
mppi_params_6['cutoff_heuristic_thr'] = 180
mppi_params_6['task_complete_thresh'] = 0.2


experiment_6 = {

    'push_world_config': config,
    'forward_model': SimpleNNModel, # Used by learnt dynamics only
    'data_path': MPPI_DATA_DIR + 'push_data_all_rand.csv',

    'is_quasi_static': True,

    'dynamics_type': 'box2d', #'learnt_dyn', 'box2d'
    'cost_type': 'cost_imp',

    'dt': mppi_params_6['dt'],

    'box2D_viewer': False,

    'random_seed': random_seed,

    'mppi_params':mppi_params_6,

    'savefig': False,

    'network_params': {
            'load_saved_model':True,
            'model_path':MPPI_DATA_DIR +'/keras_models/simple_nn_trans_push_data_all_rand.h5',
            'epochs':500,
            'batch_size':20,
            'save_model':True,
            'state_dim':2,
            'cmd_dim':mppi_params_6['cmd_dim'],
    }   
}


""" 

Experiment 7: Surface push parametarization 

"""

mppi_params_7 = copy.deepcopy(mppi_params)

#baxter data
#0.732159655755,0.276926887194,0.234879401945,0.0694952029321,0.738926599904,0.280368261557,0.264172482399
#0.717686359481,0.354216314659,0.0468301270816,0.396402005453,0.705682277723,0.419571833116,0.336554239022
#0.658223314259,0.358930527107,-0.192333118685,0.654332799806,0.662278483406,0.358735667891,-0.25399025971
#0.723033090322,0.353576904386,0.050489315064,0.814498266455,0.805259560549,0.381742693044,0.709408880844
mppi_params_7['start'] = np.array([0.717686359481,0.354216314659, 0., 0., 0., 0.])
mppi_params_7['goal'] = np.array([0.705682277723,0.419571833116, 0., 0., 0., 0.])

# mppi_params_5['traj_path'] = 'paths/follow_path_some_shape.pkl'
mppi_params_7['dynamics_type'] = 'box2d'
mppi_params_7['max_iterations'] = 2000
mppi_params_7['dt'] = 0.05
mppi_params_7['N'] = 2
mppi_params_7['K'] = 150
mppi_params_7['h'] = 1.0
mppi_params_7['rho'] = 1.0
mppi_params_7['cmd_dim'] = 1
mppi_params_7['R'] = np.eye(mppi_params_7['cmd_dim'])*0.5


mppi_params_7['goal_cost_coeff'] = 1.5
mppi_params_7['control_cost_coeff'] = 0.0
mppi_params_7['obstacle_cost_coeff'] = 0.0
mppi_params_7['vel_state_cost_coeff'] = 0.0
mppi_params_7['uncertain_cost_coeff'] = .0#4.0

mppi_params_7['use_cutoff_heuristic'] = True
mppi_params_7['cutoff_heuristic_thr'] = 180
mppi_params_7['task_complete_thresh'] = 0.2


experiment_7 = {

    'push_world_config': config,
    'forward_model': EnsambleModel, # Used by learnt dynamics only
    'data_path': MPPI_DATA_DIR + 'trans_push_data_all_rand.csv',

    'is_quasi_static': True,

    'dynamics_type': 'learnt_dyn', #'learnt_dyn', 'box2d'
    'cost_type': 'cost_imp',

    'box2D_viewer': False,

    'dt': mppi_params_7['dt'],

    'random_seed': random_seed,

    'mppi_params':mppi_params_7,

    'savefig': False,

    'network_params': {
            'train_iterations': 500,
            'n_ensembles': 10,
            'dim_input': 3, 
            'dim_output': 3,
            'n_hidden': [50,50,50,50],
            'k_mixtures': 1,
            'write_summary': False,
            'load_saved_model': False,
            'model_dir': check_point_dir,
            'model_name':'ensemble_model_u_shape.ckpt',
            'optimiser': adam_params,
            'summary_dir':summary_dir,
            'device': '/cpu:0',
            'adv_epsilon': 0.005,
    }

    

    
}


""" 

Experiment 8: Box2d real robot : ICRA PARAMS

"""

mppi_params_8 = copy.deepcopy(mppi_params)

#baxter data
#0.732159655755,0.276926887194,0.234879401945,0.0694952029321,0.738926599904,0.280368261557,0.264172482399
#0.717686359481,0.354216314659,0.0468301270816,0.396402005453,0.705682277723,0.419571833116,0.336554239022
#0.658223314259,0.358930527107,-0.192333118685,0.654332799806,0.662278483406,0.358735667891,-0.25399025971
#0.723033090322,0.353576904386,0.050489315064,0.814498266455,0.805259560549,0.381742693044,0.709408880844
# mppi_params_8['start'] = np.array([0.71, 0.482, 0.026, 0., 0., 0.])
mppi_params_8['start'] = np.array([0., 0., 0.026, 0., 0., 0.])
# mppi_params_8['start'] = np.array([0.8276729,-0.15015955, 0.0613232, 0., 0., 0.]) #facing baxter, left side start (same as goal)

# mppi_params_8['goal']  = np.array([0., -2.9, 0., 0., 0., 0.])

# mppi_params_8['goal'] = np.array([-0.16650707, -0.82253444,  0.03013755, 0., 0., 0.]) #facing baxter, left side goal but in box2d frame

#0.82264531 -0.16625783  0.02893525
#before track problem: 0.83212334, -0.16316633, 0.01281559,
# mppi_params_8['goal'] = np.array([0.82067567, -0.16507931,  0.06790948, 0., 0., 0.]) #facing baxter, left side goal
# mppi_params_8['goal'] = np.array([0.70064068, 0.76912946,  0.02922201, 0., 0., 0.]) #facing baxter, right side goal
# mppi_params_8['goal'] = np.array([0.72606981, 0.36305544,-0.06934065, 0., 0., 0.]) # 2nd row, 2nd column. kinematically better goal 1

# mppi_params_8['goal'] = np.array([0.73777145, 0.48261461, -0.00500965, 0., 0., 0.]) # 2nd row, 3rd column.
 
#this was the stuff that worked for E-MDN
# mppi_params_8['goal'] = np.array([0.70828384,  0.58655024,  1.60643172, 0., 0., 0.]) #goal on right side 1

#now the tracker shows different reading.
# 0.70414078  0.6786803   0.00545824 (1.5683167)
#0.70411289,  0.62161237, 1.5683167
#0.69396412  0.67552775  1.63927054
mppi_params_8['goal'] = np.array([0.7083, 0.5866, -0.094, 0., 0., 0.]) #goal on right side 1

mppi_params_8['obstacle'] = None #np.hstack([0.5*(mppi_params_8['start'][:2]+mppi_params_8['goal'][:2]), 0.04])
# mppi_params_8['obstacle'] = np.array([1.375, 1.375, 0.25]) #x,y, radius

mppi_params_8['max_iterations'] = 2000
mppi_params_8['dt'] = 0.05#0.009
mppi_params_8['N'] = 2 #5
mppi_params_8['K'] = 150 #150
mppi_params_8['h'] = 1.0 #1.
mppi_params_8['rho'] = 1.0
mppi_params_8['cmd_dim'] = 1
mppi_params_8['R'] = np.eye(mppi_params_8['cmd_dim'])*0.5

mppi_params_8['dynamics_type'] = 'box2d' #'box2d' 'learnt_dyn'0
mppi_params_8['goal_cost_coeff'] =  1.5 #1.5 # 10.5
mppi_params_8['goal_ori_coeff'] = 0.01 # 0.01
mppi_params_8['control_cost_coeff'] = 0.0
mppi_params_8['obstacle_cost_coeff'] = 3.5
mppi_params_8['vel_state_cost_coeff'] = 0.0
mppi_params_8['uncertain_cost_coeff'] = 0.0#4.0

mppi_params_8['use_cutoff_heuristic'] = False
mppi_params_8['cutoff_heuristic_thr'] = 180
mppi_params_8['task_complete_thresh'] = 0.07

mppi_params_8['update_rho'] =  False
mppi_params_8['rhoMin'] = 0.1
mppi_params_8['rhoMax'] = 1.1


config['dt'] = mppi_params_8['dt']
config['goal'] = mppi_params_8['goal'].copy()[:3]

adam_params = {

    'type': 'adam',
    'params': {'learning_rate' : 0.001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-08, 'use_locking': False}
}

experiment_8 = {

    'push_world_config': config,
    'forward_model': GPModel,#EnsambleModel, GPModel # Used by learnt dynamics only
    'data_path': MPPI_DATA_DIR + 'baxter_push_data_combined.csv',

    'is_quasi_static': True,

    'dynamics_type': mppi_params_8['dynamics_type'],
    'cost_type': 'cost_imp',

    'box2D_viewer': False,

    'transform_angle_baxter_2_box2d':-0.5*np.pi,

    'show_obstacle':False,

    'show_goal':True,

    'dt': mppi_params_8['dt'],

    'random_seed': random_seed,

    'mppi_params':mppi_params_8,

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
experiment_config = experiment_8





