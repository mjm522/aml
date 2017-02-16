import numpy as np

import tensorflow as tf

from aml_io.data_manager import DataManager

import matplotlib.pyplot as plt

from aml_dl.mdn.training.config import network_params_fwd, check_point_path

from aml_dl.mdn.model.nn_push_fwd_model import NNPushForwardModel

network_params_fwd['model_path'] = check_point_path + 'push_model_fwd.ckpt'


def generate_y_test(forward_model):

    data_manager = DataManager.from_file('tests/data_test.pkl')
    data_x = data_manager.pack_data_x(['state_start','state_end'])
    data_y = data_manager.pack_data_y()

    # Training ground truth

    h = forward_model.run_op('z_hidden',data_x)
    
    N_SAMPLES = 1000
    
    x_test = np.float32(np.random.uniform(-5.5, 5.5, (4, N_SAMPLES))).T
    
    h_test = forward_model.run_op('z_hidden',x_test)
    
    y_test = forward_model.sample_out(x_test, 10)

    plt.figure(figsize=(8, 8))
    plt.plot(h,data_y,'ro', h_test, y_test,'bo',alpha=0.1)
    plt.show()

    

def main():

    sess = tf.Session()

    network_params_fwd['load_saved_model'] = False

    forward_model = NNPushForwardModel(sess=sess, network_params=network_params_fwd)
    
    forward_model.init_model()


    data_manager = DataManager.from_file(network_params_fwd['training_data_path'], 
                                         network_params_fwd['train_data_file_names'])


    ids=['all']

    x_keys = ['task_before', 'task_action']

    y_keys = ['task_after']

    x_sub_keys = [['pos', 'ori'],['push_xz']]

    y_sub_keys = [['pos', 'ori']]

    data_x, data_y = data_manager.pack_sample(x_keys=x_keys, x_sub_keys=x_sub_keys, 
                                              y_keys=y_keys, y_sub_keys=y_sub_keys, 
                                              ids = ids)
    
    h = forward_model.run_op('z_hidden',data_x)

    plt.figure(figsize=(8, 8))
    plt.plot(h, data_y,'ro', alpha=0.3)
    plt.show()


    epochs = 10000#10000

    loss = forward_model.train(data_x,data_y, epochs = epochs)

    plt.figure(figsize=(8, 8))
    
    plt.plot(np.arange(100, epochs,1), loss[100:], 'r-')
    
    plt.show()

    

if __name__ == '__main__':
    main()