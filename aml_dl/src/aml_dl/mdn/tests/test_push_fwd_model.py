import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from aml_io.data_manager import DataManager
from aml_dl.mdn.model.nn_push_fwd_model import NNPushForwardModel
from aml_dl.mdn.training.config import network_params_fwd, check_point_path


def get_data(operation):

    if operation == 'test':
        data_file_name = network_params_fwd['test_data_file_names']
    elif operation =='train':
        data_file_name = network_params_fwd['train_data_file_names']

    data_manager = DataManager.from_file(network_params_fwd['training_data_path'], data_file_name) 

    ids=['all']
    x_keys = ['task_before', 'task_action']
    y_keys = ['task_after']
    x_sub_keys = [['pos', 'ori'],['push_xy']]
    y_sub_keys = [['pos', 'ori']]

    data_x, data_y = data_manager.pack_sample(x_keys=x_keys, x_sub_keys=x_sub_keys, 
                                              y_keys=y_keys, y_sub_keys=y_sub_keys, 
                                              ids = ids)

    return data_x, data_y
                                         

def test_fwd_model():

    sess = tf.Session()

    test_data_x, test_data_y = get_data('test')

    forward_model = NNPushForwardModel(sess=sess, network_params=network_params_fwd)
    forward_model.init_model()

    prediction = forward_model.run_op('z_hidden', test_data_x)

    num_outputs = network_params_fwd['dim_output']
    output_vars = network_params_fwd['output_order']

    fig, axlist = plt.subplots(num_outputs)
    d = 0
    for ax in axlist.flatten():
        ax.set_title(output_vars[d])
        h1 =  ax.plot(np.asarray(test_data_y)[:,d],    color='r', label='true')
        h2 =  ax.plot(prediction[:,d],    color='g', label='pred')
        d += 1

    fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)  # create some space below the plots by increasing the bottom-value
    axlist.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.5, -1.35), ncol=3)
    plt.suptitle('Prediction - True',size=16)
    plt.show()

    
def train_fwd_model():

    sess = tf.Session()

    network_params_fwd['load_saved_model'] = False

    forward_model = NNPushForwardModel(sess=sess, network_params=network_params_fwd)
    forward_model.init_model()

    train_data_x, train_data_y = get_data('train')
    
    # h = forward_model.run_op('z_hidden',data_x)

    # plt.figure(figsize=(8, 8))
    # plt.plot(h, data_y,'ro', alpha=0.3)
    # plt.show()

    epochs = 10000#10000
    loss = forward_model.train(train_data_x, train_data_y, epochs = epochs)

    forward_model.save_model()

    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(100, epochs,1), loss[100:], 'r-') 
    plt.show()

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and test forward model')
    parser.add_argument('-t', '--operation', type=str, help='*-t train* to Train the model or *-t test*  to Test the model')
    
    args = parser.parse_args()

    if args.operation == 'train':
        train_fwd_model()
    elif args.operation == 'test':
        test_fwd_model()
    else:
        print "Type file -h for suggestions on arguments"
  