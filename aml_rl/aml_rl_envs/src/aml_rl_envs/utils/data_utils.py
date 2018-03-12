import time
import warnings
import numpy as np

def save_csv_data(file_name, data):

    np.savetxt(file_name, data, delimiter=",")

def load_csv_data(file_name):

    return np.loadtxt(file_name, delimiter=",")