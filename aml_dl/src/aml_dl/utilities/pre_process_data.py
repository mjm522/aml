import os
import numpy as np
import tensorflow as tf
from aml_io.io_tools import save_data, load_data
from config import pre_process_siam_config
from aml_io.convert_tools import string2image
from aml_data_collec_utils.core.data_manager import DataManager
from aml_dl.mdn.training.config_exp3 import network_params_siam
from aml_dl.mdn.utilities.get_data_from_files import get_data_from_files


def get_data(data_file_indices, model_type):
    tmp_x, y_buffer = get_data_from_files(data_file_range=data_file_indices, 
                                          model_type=model_type)

    #if input is an image, then we need to convert the string to float
    if model_type == 'cnn' or model_type == 'siam':
        x_buffer = []
        for x_image in tmp_x:
            if model_type == 'siam':
                x_buffer.append((np.transpose(string2image(x_image[0][0]), axes=[2,1,0]).flatten(), 
                                       np.transpose(string2image(x_image[1][0]), axes=[2,1,0]).flatten()))
            else:
                x_buffer.append(np.transpose(string2image(x_image[0]), axes=[2,1,0]).flatten())
    else:
        x_buffer = tmp_x

    assert(len(x_buffer) == len(y_buffer))

    return x_buffer, y_buffer


def read_pre_processed_data(filename):
    data =  load_data(filename)
    return data['x'], data['y']

def input_pipeline(filenames, batch_size, num_epochs=None, shuffle=True):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=shuffle)
    x_data, y_data = read_pre_processed_data(filename_queue)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    x_batch, y_batch = tf.train.shuffle_batch([x_data, y_data], 
                                              batch_size=batch_size, 
                                              capacity=capacity,
                                              min_after_dequeue=min_after_dequeue)
    return x_batch, y_batch


def save_files():
    x_data, y_data = get_data(pre_process_siam_config['data_file_range'], pre_process_siam_config['model_type'])
    num_data_per_file = pre_process_siam_config['samples_per_file']
    file_name_prefix  = pre_process_siam_config['file_name_prefix']
    data_folder_path  = pre_process_siam_config['data_folder_path']
    under = 0
    data_file_idx = 0
    total_data = len(x_data)
    finished = False
    print "Total len", total_data
    while not finished:
        print("starting to write into queue")
        upper = under + num_data_per_file
        print("try to enqueue ", under, " to ", upper)
        if upper <= total_data:
            curr_data = x_data[under:upper]
            curr_target = y_data[under:upper]
            under = upper
        else:
            rest = upper - total_data
            curr_data = np.concatenate((x_data[under:total_data], x_data[0:rest]))
            curr_target = np.concatenate((y_data[under:total_data], y_data[0:rest]))
            under = rest
            finished = True

        data = {'x':curr_data, 'y':curr_target}
        data_file_idx += 1
        print "Len of x", len(curr_data)
        print "Len of y", len(curr_target)
        filename = data_folder_path + file_name_prefix + '_%02d.pkl'%data_file_idx
        save_data(data=data, filename=filename)


def main(): 
    save_files()
    # data_folder_path  = pre_process_siam_config['data_folder_path']
    # file_name_prefix  = pre_process_siam_config['file_name_prefix']
    # filenames = [data_folder_path + file_name_prefix + '_%02d.pkl'%idx for idx in range(1,5)]
    # x_batch, y_batch = input_pipeline(filenames=filenames, batch_size=10)

    # print "Len of batch \t", len(x_batch)
    # print "Len of batch \t", len(y_batch)

if __name__ == '__main__':
    main()