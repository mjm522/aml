import os
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from aml_io.io_tools import load_data

types=['front', 'top', 'front_left', 'front_left_top', 'front_right', 'front_right_top', 'left', 'right']

data_file_base = os.environ['AML_DATA'] + '/aml_playground/sawyer_spring/new_spring_demos/%s_0%d.pkl'

def lpf(data):
    b, a = sig.butter(3, 0.05)
    return sig.filtfilt(b, a, data)

def run_avg(data, N=None):
    if N is None:
        N =  len(data)
    return np.convolve(data, np.ones((N,))/N, mode='valid')

def save_data(file_name):
    
    plt.ion()

    X_train = None

    for i in range(1,6):

        data = load_data( data_file_base%(file_name, i) )[0]
   
        force_data  = []
        torque_data = []
        ee_pos_data = []
        ee_vel_data = []
        ee_ori_data = []
        ee_omg_data = []
        js_pos_data = []

        for k in range(data.size):
            force_data.append(data.get(k, ['ft_reading'])[:3])
            torque_data.append(data.get(k, ['ft_reading'])[3:])
            ee_pos_data.append(data.get(k, ['ee_point']))
            ee_vel_data.append(data.get(k, ['ee_vel']))
            ee_ori_data.append(data.get(k, ['ee_ori']))
            ee_omg_data.append(data.get(k, ['ee_omg']))
            js_pos_data.append(data.get(k, ['position']))

        force_data = np.asarray(force_data).squeeze()
        ee_pos_data = np.asarray(ee_pos_data).squeeze()
        js_pos_data = np.asarray(js_pos_data).squeeze()
        ee_vel_data = np.asarray(ee_vel_data).squeeze()

        vals = range(0, data.size)

        plt.figure('force')
        plt.subplot(2,1,1)
        plt.plot(force_data[:,0], 'r')
        plt.plot(force_data[:,1], 'g')
        plt.plot(force_data[:,2], 'b')
        plt.subplot(2,1,2)
        plt.plot(lpf(force_data[vals,0]), 'r')
        plt.plot(lpf(force_data[vals,1]), 'g')
        plt.plot(lpf(force_data[vals,2]), 'b')
        plt.draw()
        plt.pause(0.00001)

        plt.figure('pos')
        plt.subplot(2,1,1)
        plt.plot(ee_pos_data[:,0], 'r')
        plt.plot(ee_pos_data[:,1], 'g')
        plt.plot(ee_pos_data[:,2], 'b')
        plt.subplot(2,1,2)
        plt.plot(run_avg(ee_pos_data[vals,0], 50), 'r')
        plt.plot(run_avg(ee_pos_data[vals,1], 50), 'g')
        plt.plot(run_avg(ee_pos_data[vals,2], 50), 'b')
        plt.draw()
        plt.pause(0.00001)

        force_data_tmp = np.vstack([lpf(force_data[vals,0]), 
                                    lpf(force_data[vals,1]),
                                    lpf(force_data[vals,2])]).T

        pos_data_tmp = np.vstack([run_avg(ee_pos_data[vals,0], 50), 
                                  run_avg(ee_pos_data[vals,1], 50),
                                  run_avg(ee_pos_data[vals,2], 50) ]).T

        vel_data_tmp = np.vstack([run_avg(ee_vel_data[vals,0], 50), 
                                  run_avg(ee_vel_data[vals,1], 50),
                                  run_avg(ee_vel_data[vals,2], 50) ]).T

        
        pos_shape = pos_data_tmp.shape
        vel_shape = vel_data_tmp.shape
        force_shape = force_data_tmp.shape

        min_ = min(pos_shape[0], vel_shape[0], force_shape[0])

        datum = np.hstack([ pos_data_tmp[:min_,:], vel_data_tmp[:min_,:], force_data_tmp[:min_,:] ])

        # datum = np.hstack([ ee_pos_data,
        #                     ee_vel_data,
        #                     np.vstack( [lpf(force_data[vals,0])],
        #                                [lpf(force_data[vals,1])]
        #                                [lpf(force_data[vals,2])] ).T ])


        # print datum.shape
        # raw_input()

        
        if X_train is None:
            X_train = datum
        else:
            X_train = np.vstack([X_train, datum])

    np.savetxt('train_data_%s.npy'%file_name, X_train)
    print X_train.shape
    # raw_input()

def batch_process():

    save_path = os.environ['AML_DATA'] + '/aml_playground/sawyer_spring/new_spring_demos/pre_processed/'

    datum = None

    def rearange(_data):
        num_data = _data.shape[0]
        x_data   = _data[0:num_data-1, :]
        ft1_data = _data[1:num_data, 6:]
        new_data = np.hstack([x_data, ft1_data])
        return new_data


    for file_name in types:

        print  "Type of file: %s"%file_name

        load_data = np.loadtxt(save_path+'train_data_%s.npy'%file_name)

        if datum is None:

            datum = rearange(load_data)

        else:

            datum = np.vstack([datum, rearange(load_data)])

    np.savetxt(save_path+'train_data_total_with_ft1.npy', datum)

    print datum.shape
        # save_data(file_name)


if __name__ == '__main__':

    batch_process()

    raw_input()

