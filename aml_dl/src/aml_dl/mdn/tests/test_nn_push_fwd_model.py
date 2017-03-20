import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from aml_dl.mdn.training.config import network_params_fwd
from aml_dl.utilities.tf_batch_creator import BatchCreator
from aml_dl.mdn.model.nn_push_fwd_model import NNPushFwdModel
from aml_dl.mdn.utilities.get_data_from_files import get_data_from_files

def get_data(operation):

    if operation == 'test':
        data_file_indices = network_params_fwd['test_file_indices']
    elif operation =='train':
        data_file_indices = network_params_fwd['train_file_indices']

    data_x, data_y = get_data_from_files(data_file_range=data_file_indices, model_type='fwd')

    return data_x, data_y
                               

def test_fwd_model():

    if network_params_fwd['write_summary']:
        sess = tf.InteractiveSession()
    else:
        sess = tf.Session()

    test_data_x, test_data_y = get_data('test')

    print "Got the data, gonna test the model..."

    forward_model = NNPushFwdModel(sess=sess, network_params=network_params_fwd)
    forward_model.init_model()

    prediction = forward_model.run_op('output', test_data_x)

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
    fig.subplots_adjust(hspace=.5)
    plt.show()

    
def train_fwd_model():

    if network_params_fwd['write_summary']:
        sess = tf.InteractiveSession()
    else:
        sess = tf.Session()

    train_data_x = None; train_data_y = None; batch_creator = None   
    if network_params_fwd['batch_params'] is not None:
        batch_creator = BatchCreator(network_params_fwd['batch_params'])
    else:
        train_data_x, train_data_y = get_data('train')

    network_params_fwd['load_saved_model'] = False

    forward_model = NNPushFwdModel(sess=sess, network_params=network_params_fwd)
    forward_model.init_model()
    forward_model.configure_data(data_x=train_data_x, data_y=train_data_y, batch_creator=batch_creator)
    
    # h = forward_model.run_op('last_hidden',data_x)

    # plt.figure(figsize=(8, 8))
    # plt.plot(h, data_y,'ro', alpha=0.3)
    # plt.show()

    print "Got the data, gonna train the model..."

    epochs = 100#10000
    loss = forward_model.train(epochs = epochs)
    forward_model.save_model()
    
    if network_params_fwd['write_summary']:
        logdir = forward_model._tf_sumry_wrtr._summary_dir
        instruction = 'tensorboard --logdir=' + logdir
        os.system(instruction)

    else:
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
  