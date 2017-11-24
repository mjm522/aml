import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm


def sigmoid(x):
    return 1./(1.+np.exp(-x))


def get_circle(radius, position, steps = 100):
    #for plotting obstacles
    th = np.linspace(0., 2*np.pi, steps)
    x_obs = position[0] + radius*np.cos(th)
    y_obs = position[1] + radius*np.sin(th)


    return x_obs, y_obs