import os


#######################################################################################################
## configuration for Enesemble model
check_point_dir   = 'tf_check_points/exp_ensemble/'
summary_dir = 'exp_ensemble/'

adam_params = {
    'type': 'adam',
    'params': {'learning_rate' : 0.0001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-08, 'use_locking': False}
}

network_params =  {
            'train_iterations': 2500,
            'n_ensembles': 10,
            'dim_input': 5, 
            'dim_output': 3,
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

ensemble_config = {'network_params': network_params,
          'random_seed':0,}

###################################################################################################


dropout_config = {
    'load_saved_model':False,
    'save_model':True,
    'model_path': os.environ['MPPI_DATA_DIR'] + 'keras_models' + '/1000_epochs_all_gap.h5',
    'Ns':[390],#np.array([10, 25, 50, 100, 390]), #np.array([390]),#
    'nb_epochs':[1000],#[2000, 1000, 500, 200, 20, 2], ##,
    'nb_val_size':100,#1000,
    'nb_features':50,
    'Q':3,
    'D':3,
    'K_test':50,
    'nb_reps':1,#3
    'batch_size':20,
    'l':1e-4,
    'optimizer':'adam', #'sgd', 'adam'
}


