import numpy as np
from aml_data_collec_utils.core.data_manager import DataManager
from aml_dl.mdn.training.config2 import network_params_inv, network_params_fwd, network_params_cmbnd, network_params_siam

def get_data_from_files(data_file_range, model_type, ids=None):
        
        if model_type == 'fwd':

            data_man = DataManager(data_folder_path=network_params_inv['training_data_dir'], data_name_prefix='test_push_data')

            x_keys = ['box_pos', 'box_ori', 'task_action']
            y_keys = ['box_pos', 'box_ori']
            x_sub_keys = [[None],[None],['push_xz']]
            y_sub_keys = [[None],[None]]

            x_sample_points = [0]
            y_sample_points = [0]


        elif model_type == 'inv':

            data_man = DataManager(data_folder_path=network_params_fwd['training_data_dir'], data_name_prefix='test_push_data')  
            
            y_keys = ['task_action']
            x_keys = ['box_pos', 'box_ori']
            x_sub_keys = [[None],[None]]
            y_sub_keys = [['push_xz']]

            x_sample_points = [0,-1]
            y_sample_points = [0]

        elif model_type == 'cnn':

            data_man = DataManager(data_folder_path=network_params_cmbnd['training_data_dir'], data_name_prefix='test_push_data')

            x_keys = ['rgb_image']
            x_sub_keys = None
            y_keys = ['box_pos', 'box_ori']
            y_sub_keys = [[None],[None]]

            x_sample_points = [0]
            y_sample_points = [0]

        elif model_type == 'siam':
            print network_params_siam['training_data_dir']
            print "*****************************"
            data_man = DataManager(data_folder_path=network_params_siam['training_data_dir'], data_name_prefix='test_push_data')

            x_keys = ['rgb_image']
            x_sub_keys = None
            #this is a hack, so we can seperate action later,
            #it is not easy to stack it with the rgb image since 
            #it is a string.
            y_keys = ['box_pos', 'box_ori', 'task_action']
            y_sub_keys = [[None],[None], ['push_xz']]


            x_sample_points = [0,-1]
            y_sample_points = [0,-1]


        data_x = data_man.pack_data_in_range(keys=x_keys, 
                                             sub_keys=x_sub_keys, 
                                             ids=ids, 
                                             sample_points=x_sample_points, 
                                             data_file_range=data_file_range)
        
        data_y = data_man.pack_data_in_range(keys=y_keys, 
                                             sub_keys=y_sub_keys, 
                                             ids=ids, 
                                             sample_points=y_sample_points, 
                                             data_file_range=data_file_range)
            


        return data_x, data_y