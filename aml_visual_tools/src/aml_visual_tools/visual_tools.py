import cv2
import random
import numpy as np
#TODO run the plotting on seperate core.
# import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from mpl_toolkits.mplot3d import Axes3D
from aml_io.convert_tools import string2image, rosimage2openCVimage

import rospy

def visualize_3D_data(data, fig_handle=None, axis_lim=None, color=None, label=None):
    '''
    expects list of lists with x, y, z are the order of lists
    i.e. data = [data_x, data_y, data_z], numpy array can also be passed shape: (3xN)
    axis_lim is of the form [x_min, x_max, y_min, y_max]
    '''
    if isinstance(data, np.ndarray):
        data = data.tolist()

    if color is None:
        color = 'g'

    if label is None:
        label = 'data'

    if fig_handle is None:
        fig = plt.figure()
    else:
        fig = fig_handle

    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data[0], data[1], data[2], color=color, label=label)
    if axis_lim is not None:
        ax.axis(axis_lim)
    plt.show()
    return fig

def visualize_2D_data(data, fig_handle=None, axis_lim=None, color=None, label=None):
    '''
    expects list of lists with x, y are the order of lists
    i.e. data = [data_x, data_y], numpy array can also be passed shape: (2xN)/(1xN)
    axis_lim is of the form [x_min, x_max, y_min, y_max]
    '''
    if isinstance(data, np.ndarray):
        data = data.tolist()

    if len(data) == 1:
        data = [range(len(data)), data]

    if color is None:
        color = 'g'

    if label is None:
        label = 'data'

    if fig_handle is None:
        fig = plt.figure()
    else:
        fig = fig_handle
    ax = fig.add_subplot(111)
    
    ax.plot(data[0], data[1], color=color, label=label)

    if axis_lim is not None:
        ax.axis(axis_lim)
    plt.show()
    return fig

def visualize_2D_data_with_sigma(data, sigma, stddev=3.,fig_handle=None, axis_lim=None, color=None, label=None):
    '''
    expects numpy array to be passed, shape: (2xN)
    sigma is an array of dimention 1xN
    axis_lim is of the form [x_min, x_max, y_min, y_max]
    '''
    if isinstance(data, list):
        data = np.asarray(data)

    if data.shape[0] == 1:
        data = np.vstack([range(data.shape[1]), data])

    print data.shape

    if color is None:
        color = 'g'

    if label is None:
        label = 'data'

    if isinstance(sigma, list):
        sigma = np.asarray(sigma)

    if fig_handle is None:
        fig = plt.figure()
    else:
        fig = fig_handle
    ax = fig.add_subplot(111)

    x   = data[0]
    ypl = data[1]-stddev*sigma
    yph = data[1]+stddev*sigma

    ax.fill_between(x.tolist(), ypl.tolist()[0], yph.tolist()[0], facecolor='grey', interpolate=True, alpha=0.5)
    ax.plot(x, data[1], color=color, label=label) 
    ax.legend(('mu','variance'))
    if axis_lim is not None:
        ax.axis(axis_lim)
    ax.grid()
    plt.show()
    return fig

def continous_3D_plot(data, fig_handle=None, axis_lim=None, color=None):
    if isinstance(data, np.ndarray):
        data = data.tolist()

    if color is None:
        color = 'g'

    if fig_handle is None:
        fig = plt.figure()
        plt.ion()
        plt.show()
        X = []
        Y = []
        Z = []
        rospy.Timer(0.001, partial(update_plot, X, Y, Z, color))

    else:
        fig = fig_handle

    X.append(data[0])
    Y.append(data[1])
    Z.append(data[2])

    def update_plot(X, Y, Z, color, event):
        while not rospy.is_shutdown():
            fig.gca(projection='3d').plot(X, Y, Z, color)
            plt.pause(0.0001)
            plt.draw()

    return fig

def continous_2D_plot(data, X = [], Y = [], fig_handle=None, axis_lim=None, color=None):
    if isinstance(data, np.ndarray):
        data = data.tolist()

    if color is None:
        color = 'g'




    def update_plot(X, Y, event):
        while not rospy.is_shutdown():
            fig.add_subplot(111).plot(X, Y, color)
            plt.pause(0.0001)
            plt.draw()

    if fig_handle is None:    
        fig = plt.figure()
        plt.ion()
        plt.show()
        rospy.Timer(rospy.Duration(0.001), partial(update_plot, X, Y))
    else:
        fig = fig_handle

    X.append(data[0])
    Y.append(data[1])


    return fig, X, Y


def show_image(image_data, window_name=None):

    if window_name is None:
        window_name = 'Show image window'

    if isinstance(image_data, Image):
        image = rosimage2openCVimage(image_data)
    elif isinstance(image_data[0], str):
        image = string2image(image_data[0])
    else:
        image = image_data

    cv2.imshow(window_name, image)
    cv2.waitKey(0)


def visualize_multiple_subplots(data, fig_handle=None, color=None, label=None, ax_title=None, fig_title='Data'):

    if isinstance(data, np.ndarray):
        data = data.tolist()

    if fig_handle is None:
        fig = plt.figure()
    else:
        fig = fig_handle
    ax = fig.add_subplot(111)

    num_subplots = len(data)

    if color is None:
        color = [[random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)] for _ in range(num_subplots)]

    if label is None:
        label = ['data%1d'%k for k in range(num_subplots)]

    if ax_title is None:
        ax_title = label

    print ax_title
    print label
    
    fig, axlist = plt.subplots(num_subplots)
    print len(axlist.flatten())
    print num_subplots
    for ax, k in zip(axlist.flatten(),range(num_subplots)):
        ax.set_title(ax_title[k])
        ax.plot(data[k], color=color[k], label=label[k])

    fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)  # create some space below the plots by increasing the bottom-value
    axlist.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.5, -1.35), ncol=3)
    plt.suptitle(fig_title, size=16)
    fig.subplots_adjust(hspace=.5)
    plt.show()
    return fig