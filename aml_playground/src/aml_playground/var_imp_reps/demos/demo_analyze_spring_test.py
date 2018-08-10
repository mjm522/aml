import os
import copy
import pandas
import numpy as np
import quaternion as q
import matplotlib.pyplot as plt
from rl_algos.utils.utils import lpf
from aml_io.io_tools import load_data
from scipy.interpolate import interp1d


file_path = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/creps_data_only_x_after_fix_2_extern_Aug_10_2018_04_02_12.pkl'


def xyz_plot(data, labels = None, axis_labels = None, title = None, multiplot = None):
    ''' 

    Args : 
        # data : list of np.arrays of shape [:,3]
        # labels : corresponding list of triples for labels. Eg. (['px','py','pz'],['kx','ky','kz'],...)
        # title : plot title
        # multiplot : for overlaying one plot on another, give multiplot = { col_num : [data1, data2, ..], ... } 

    '''

    num_cols = len(data)

    labelling = False

    if labels is not None:
        labelling = True
        assert len(labels) == num_cols

    if title is not None:
        plt.figure(title)

    multiplotting = False
    if multiplot is not None:
        multiplotting = True

    for i in range(num_cols):

        plt.subplot(3, num_cols, i + 1)
        plt.plot(data[i][:,0])

        if axis_labels is not None:
            plt.xlabel(axis_labels[i][0][0])
            plt.ylabel(axis_labels[i][0][1])

        if multiplotting == True and i+1 in multiplot.keys():
            for j in range(len(multiplot[i+1])):
                plt.plot(multiplot[i+1][j][:,0])
        if labelling:
            plt.title(labels[i][0])

        plt.subplot(3, num_cols, i + num_cols + 1)
        plt.plot(data[i][:,1])

        if axis_labels is not None:
            plt.xlabel(axis_labels[i][1][0])
            plt.ylabel(axis_labels[i][1][1])

        if multiplotting == True and i+1 in multiplot.keys():
            for j in range(len(multiplot[i+1])):
                plt.plot(multiplot[i+1][j][:,1])
        if labelling:
            plt.title(labels[i][1])

        plt.subplot(3, num_cols, i + 2*num_cols + 1)
        plt.plot(data[i][:,2])
        
        if axis_labels is not None:
            plt.xlabel(axis_labels[i][2][0])
            plt.ylabel(axis_labels[i][2][1])

        if multiplotting == True and i+1 in multiplot.keys():
            for j in range(len(multiplot[i+1])):
                plt.plot(multiplot[i+1][j][:,2])
        if labelling:
            plt.title(labels[i][2])



def visualise_data(show_plots = True): #right2left left2right

    data = load_data(file_path)
    
    plt.ion()

    mean_reward = [data[i]['mean_reward']  for i in range(len(data))]

    for i in range(len(data)):

        ee_traj = data[i]['ee_traj']
        req_traj = data[i]['traj']
        w_list = np.asarray(data[i]['params'])

        plotdata = [ee_traj, w_list[:,:3], w_list[:,3:]]
        labels = [['ee_x','ee_y','ee_z'], ['kp_x','kp_y','kp_z'], ['kd_x','kd_y','kd_z']]
        axis_labels = None
        
        xyz_plot(data = plotdata, labels = labels, axis_labels=axis_labels, multiplot = { 1 : [req_traj] } ,title = "Test Impedance Learning")

        print i

        plt.draw()
        plt.pause(0.0001)

        raw_input()

    # plt.plot(mean_reward)
    # plt.show()
    # raw_input('adf')



if __name__ == '__main__':


    visualise_data()
