import os
import argparse
import numpy as np
import multiprocessing
import tensorflow as tf
import matplotlib.pyplot as plt
from aml_io.convert_tools import string2image
from aml_dl.utilities.tf_batch_creator import BatchCreator
from aml_dl.mdn.training.config_debug import network_params_siam
from aml_dl.mdn.model.siamese_push_model import SiamesePushModel
from aml_dl.mdn.utilities.get_data_from_files import get_data_from_files


def get_session():
    if network_params_siam['device'] == '/cpu:0':
        config=tf.ConfigProto(intra_op_parallelism_threads=multiprocessing.cpu_count())
    elif network_params_siam['device'] == '/gpu:0':
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    if network_params_siam['write_summary']:
        sess = tf.InteractiveSession(config=config)
    else:
        sess = tf.Session(config=config)
    return sess


def get_data(operation, string_img_convert=True):

    raise Exception("Fix this damn thing")

    if operation == 'test':
        data_file_indices = network_params_siam['test_file_indices']
    elif operation =='train':
        data_file_indices = network_params_siam['train_file_indices']

    tmp_x, data_y = get_data_from_files(data_file_range=data_file_indices, model_type='siam')

    data_x = []
    if string_img_convert:
        for x_image in tmp_x:
            data_x.append((np.transpose(string2image(x_image[0][0]), axes=[2,1,0]).flatten(), 
                           np.transpose(string2image(x_image[1][0]), axes=[2,1,0]).flatten()))

    else:
        data_x = tmp_x

    return np.asarray(data_x), np.asarray(data_y)

def test_siamese_model():

    network_params_siam['load_saved_model'] = True

    test_data_x = None; test_data_y = None; batch_creator = None   
    if network_params_siam['batch_params'] is not None:

        network_params_siam['batch_params']['data_file_indices'] = network_params_siam['batch_params']['test_data_file_indices']

        batch_creator = BatchCreator(network_params_siam['batch_params'])
    else:
        test_data_x, test_data_y = get_data('test')

    siamese_model = SiamesePushModel(sess=get_session(), network_params=network_params_siam)
    siamese_model.init_model(epoch = 560)
    siamese_model.configure_data(data_x=test_data_x, data_y=test_data_y, batch_creator=batch_creator)
    
    # h = siamese_model.run_op('last_hidden',data_x)
    # plt.figure(figsize=(8, 8))
    # plt.plot(h, data_y,'ro', alpha=0.3)
    # plt.show()

    print "Got the data, gonna train the model..."

    epochs = network_params_siam['epochs']
    siamese_model.test(iterations=epochs)


    
def train_siamese_model():

    network_params_siam['load_saved_model'] = False

    train_data_x = None; train_data_y = None; batch_creator = None   
    if network_params_siam['batch_params'] is not None:
        batch_creator = BatchCreator(network_params_siam['batch_params'])
    else:
        train_data_x, train_data_y = get_data('train')

    siamese_model = SiamesePushModel(sess=get_session(), network_params=network_params_siam)
    siamese_model.init_model(epoch = 50)
    siamese_model.configure_data(data_x=train_data_x, data_y=train_data_y, batch_creator=batch_creator)
    
    # h = siamese_model.run_op('last_hidden',data_x)
    # plt.figure(figsize=(8, 8))
    # plt.plot(h, data_y,'ro', alpha=0.3)
    # plt.show()

    print "Got the data, gonna train the model..."

    epochs = network_params_siam['epochs']
    siamese_model.train2(iterations=epochs, chk_pnt_save_invl=network_params_siam['check_point_save_interval'])

    siamese_model.save_model('final') #providing subscript for the final model saved, so the model name is final_blah_lah

    if network_params_siam['write_summary']:
        logdir = siamese_model._tf_sumry_wrtr._summary_dir
        instruction = 'tensorboard --logdir=' + logdir
        os.system(instruction)

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and test forward model')
    parser.add_argument('-t', '--operation', type=str, help='*-t train* to Train the model or *-t test*  to Test the model')
    
    args = parser.parse_args()

    if args.operation == 'train':
        train_siamese_model()
    elif args.operation == 'test':
        test_siamese_model()
    else:
        print "Type file -h for suggestions on arguments"


