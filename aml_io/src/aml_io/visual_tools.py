import cv2
import numpy as np
#TODO run the plotting on seperate core.
# import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from aml_io.convert_tools import string2image
from cv_bridge import CvBridge, CvBridgeError


def visualize_3D_data(data, fig_handle=None, axis_lim=None):
    if fig_handle is None:
        fig = plt.figure()
    else:
        fig = fig_handle

    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data[0], data[1], data[2])
    plt.show()
    return fig


def visualize_2D_data(data, fig_handle=None, axis_lim=None):
    if fig_handle is None:
        fig = plt.figure()
    else:
        fig = fig_handle
    ax = fig.add_subplot(111)
    ax.plot(data[0], data[1])
    plt.show()
    return fig


def continous_3D_plot(data, fig_handle=None, axis_lim=None):
    if fig_handle is None:
        fig = plt.figure()
        plt.ion()
        plt.show()
        X = []
        Y = []
        Z = []
        rospy.Timer(0.001, partial(update_plot, X, Y, Z))

    else:
        fig = fig_handle

    X.append(data[0])
    Y.append(data[1])
    Z.append(data[2])

    def update_plot(X, Y, Z, event):
        while not rospy.is_shutdown():
            fig.gca(projection='3d').plot(X, Y, Z)
            plt.pause(0.0001)
            plt.draw()

    return fig

def continous_2D_plot(data, fig_handle=None, axis_lim=None):  
    if fig_handle is None:    
        fig = plt.figure()
        plt.ion()
        plt.show()
        X = []
        Y = []
        rospy.Timer(0.001, partial(update_plot, X, Y))
    else:
        fig = fig_handle

    X.append(data[0])
    Y.append(data[1])

    def update_plot(X, Y, event):
        while not rospy.is_shutdown():
            fig.add_subplot(111).plot(X, Y)
            plt.pause(0.0001)
            plt.draw()

    return fig


def show_image(image_data, window_name=None):

    if window_name is None:
        window_name = 'Show image window'

    if isinstance(image_data[0], str):
        image = string2image(image_data[0])
    else:
        image = image_data

    cv2.imshow(window_name, image)
    cv2.waitKey(0)