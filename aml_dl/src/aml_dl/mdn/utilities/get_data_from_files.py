import numpy as np
from aml_dl.mdn.training.config import network_params_inv, network_params_fwd
from aml_data_collec_utils.core.data_manager import DataManager

def get_data_from_files(data_file_range, model_type):
    
    ids=range(0,5)

    if model_type == 'fwd':

        data_man = DataManager(data_folder_path=network_params_inv['training_data_path'], data_name_prefix='test_push_data')

        x_keys = ['box_pos', 'box_ori', 'task_action']
        y_keys = ['box_pos', 'box_ori']
        x_sub_keys = [[None],[None],['push_xz']]
        y_sub_keys = [[None],[None]]

        data_x = data_man.pack_data_in_range(keys=x_keys, 
                                             sub_keys=x_sub_keys, 
                                             ids=ids, 
                                             sample_points=[0], 
                                             data_file_range=data_file_range)
        
        data_y = data_man.pack_data_in_range(keys=y_keys, 
                                              sub_keys=y_sub_keys, 
                                              ids=ids, 
                                              sample_points=[0], 
                                              data_file_range=data_file_range)

    elif model_type == 'inv':

        data_man = DataManager(data_folder_path=network_params_fwd['training_data_path'], data_name_prefix='test_push_data')  
        
        y_keys = ['task_action']
        x_keys = ['box_pos', 'box_ori']
        x_sub_keys = [[None],[None]]
        y_sub_keys = [['push_xz']]

        data_x1 = data_man.pack_data_in_range(keys=x_keys, 
                                              sub_keys=x_sub_keys, 
                                              ids=ids, 
                                              sample_points=[0], 
                                              data_file_range=data_file_range)
        
        data_x2 = data_man.pack_data_in_range(keys=x_keys, 
                                             sub_keys=x_sub_keys, 
                                             ids=ids, 
                                             sample_points=[-1], 
                                             data_file_range=data_file_range)
        
        data_y  = data_man.pack_data_in_range(keys=y_keys, 
                                              sub_keys=y_sub_keys, 
                                              ids=ids,  
                                              sample_points=[0], 
                                              data_file_range=data_file_range)

        data_x  = np.hstack([np.asarray(data_x1),np.asarray(data_x2)]).tolist()
    


    return data_x, data_y