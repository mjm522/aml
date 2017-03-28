import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from aml_data_collec_utils.core.data_manager import DataManager
from aml_dl.mdn.model.mdn_push_inv_model import MDNPushInverseModel
from aml_visual_tools.visual_tools import visualize_2D_data_with_sigma
from aml_dl.mdn.utilities.get_data_from_files import get_data_from_files
from aml_dl.mdn.training.config import network_params_inv, check_point_path


def get_data(operation):

    if operation == 'test':
        data_file_indices = network_params_inv['test_file_indices']
    elif operation =='train':
        data_file_indices = network_params_inv['train_file_indices']

    data_x, data_y = get_data_from_files(data_file_range=data_file_indices, model_type='inv')

    return data_x, data_y
                                         

def test_inv_model():

    if network_params_inv['write_summary']:
        sess = tf.InteractiveSession()
    else:
        sess = tf.Session()

    test_data_x, test_data_y = get_data('test')

    print "Got the data, gonna test the model..."

    inverse_model = MDNPushInverseModel(sess=sess, network_params=network_params_inv)
    inverse_model.init_model()

    mu_out = inverse_model.run_op('mu', test_data_x)
    sigma_out = inverse_model.run_op('sigma', test_data_x)
    pi_out = inverse_model.run_op('pi', test_data_x)


    max_ids = map(inverse_model._max_pi_idx, pi_out)
    
    pred_mu = []
    for i in range(len(max_ids)):
        pred_mu.append(mu_out[i,:,max_ids[i]])

    pred_sigma = []
    for i in range(len(max_ids)):
        pred_sigma.append(sigma_out[i,max_ids[i]])
    
    # print len(max_ids)
    # print pred_mu[0].shape
    # print len(pred_sigma)

    test_data_y = np.asarray(test_data_y).T
    pred_mu = np.asarray(pred_mu).T
    #summing all varainces together!
    pred_sigma = np.asarray(pred_sigma)[None,:]

    print pred_sigma.shape
    print test_data_y.shape

    num_outputs = network_params_inv['dim_output']
    output_vars = ['x','y']

    visualize_2D_data_with_sigma(data=test_data_y[0][None,:], sigma=pred_sigma)
    visualize_2D_data_with_sigma(data=test_data_y[1][None,:], sigma=pred_sigma)

    
def train_inv_model():

    if network_params_inv['write_summary']:
        sess = tf.InteractiveSession()
    else:
        sess = tf.Session()

    network_params_inv['load_saved_model'] = False

    inverse_model = MDNPushInverseModel(sess=sess, network_params=network_params_inv)
    inverse_model.init_model()

    train_data_x, train_data_y = get_data('train')
    
    # h = inverse_model.run_op('last_hidden',data_x)

    # plt.figure(figsize=(8, 8))
    # plt.plot(h, data_y,'ro', alpha=0.3)
    # plt.show()

    print "Got the data, gonna train the model..."

    epochs = 100000#10000

    loss = inverse_model.train(train_data_x, train_data_y, epochs = epochs)

    inverse_model.save_model()

    if network_params_inv['write_summary']:

        plt.figure(figsize=(8, 8))
        plt.plot(np.arange(100, epochs,1), loss[100:], 'r-') 
        plt.show()

        logdir = inverse_model._tf_sumry_wrtr._summary_dir
        instruction = 'tensorboard --logdir=' + logdir
        os.system(instruction)
    else:
        plt.figure(figsize=(8, 8))
        plt.plot(np.arange(100, epochs,1), loss[100:], 'r-') 
        plt.show()

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and test inverse model')
    parser.add_argument('-t', '--operation', type=str, help='*-t train* to Train the model or *-t test*  to Test the model')
    
    args = parser.parse_args()

    if args.operation == 'train':
        train_inv_model()
    elif args.operation == 'test':
        test_inv_model()
    else:
        print "Type file -h for suggestions on arguments"
  