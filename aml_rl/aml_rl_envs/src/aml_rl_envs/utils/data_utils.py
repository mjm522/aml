import os
import time
import warnings
import numpy as np
from aml_io.log_utils import aml_logging

def save_csv_data(file_name, data):

    path_name = '/'.join(file_name.split('/')[:-1])

    if not os.path.exists(path_name):
        aml_logging.warning('Path %s not found, creating new.'%(path_name,))
        os.makedirs(path_name)

    np.savetxt(file_name, data, delimiter=",")

def load_csv_data(file_name):

    return np.loadtxt(file_name, delimiter=",")