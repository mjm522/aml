import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from aml_io.convert_tools import string2image
from aml_data_collec_utils.core.data_manager import DataManager
from aml_dl.mdn.model.cnn_model_with_summary import CNNModelWithSummary
from aml_dl.mdn.training.config import network_params_cmbnd


def get_data(operation, string_img_convert=True):

    if operation == 'test':
        data_file_indices = network_params_cmbnd['test_file_indices']
    elif operation =='train':
        data_file_indices = network_params_cmbnd['train_file_indices']

    data_man = DataManager(data_folder_path=network_params_cmbnd['training_data_path'], data_name_prefix='test_push_data')

    ids=range(0,5)
    x_keys = ['rgb_image']
    x_sub_keys = None
    y_keys = ['box_pos', 'box_ori']
    y_sub_keys = [[None],[None]]

    tmp_x, data_y = data_man.pack_data_in_range_xy(x_keys=x_keys, y_keys=y_keys, 
                                                    x_sub_keys=x_sub_keys, y_sub_keys=y_sub_keys, 
                                                    ids=ids, 
                                                    before_after=True, 
                                                    data_file_range=data_file_indices)

    data_x = []
    if string_img_convert:
        for x_image in tmp_x:
            data_x.append(string2image(x_image[0]).flatten())
    else:
        data_x = tmp_x

    return np.asarray(data_x), np.asarray(data_y)


def test_fwd_model():

    if network_params_cmbnd['write_summary']:
        sess = tf.InteractiveSession()
    else:
        sess = tf.Session()

    test_data_x, test_data_y = get_data('test')

    print "Got the data, gonna test the model..."

    forward_model = CNNModelWithSummary(sess=sess, network_params=network_params_cmbnd)
    forward_model.init_model()

    prediction = forward_model.run_op('output', test_data_x)

    num_outputs = network_params_cmbnd['dim_output']
    output_vars = network_params_cmbnd['output_order']

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

    if network_params_cmbnd['write_summary']:
        sess = tf.InteractiveSession()
    else:
        sess = tf.Session()

    network_params_cmbnd['load_saved_model'] = False

    forward_model = CNNModelWithSummary(sess=sess, network_params=network_params_cmbnd)
    forward_model.init_model()

    train_data_x, train_data_y = get_data('train')
    
    # h = forward_model.run_op('last_hidden',data_x)

    # plt.figure(figsize=(8, 8))
    # plt.plot(h, data_y,'ro', alpha=0.3)
    # plt.show()

    print "Got the data, gonna train the model..."

    epochs = 100#10000
    forward_model.train(train_data_x, train_data_y, epochs = epochs)

    forward_model.save_model()

    if network_params_cmbnd['write_summary']:
        logdir = forward_model._tf_sumry_wrtr._summary_dir
        instruction = 'tensorboard --logdir=' + logdir
        os.system(instruction)

    
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