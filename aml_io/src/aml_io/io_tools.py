import time
import pickle
import warnings
from os.path import exists, join, dirname, abspath

def get_aml_package_path(aml_package_name=None):

    #if the package name is none, it returns path to aml folder
    if aml_package_name is None:

        aml_package_path = '/'.join(dirname(dirname(abspath(__file__))).split('/')[:-2])

    else:

        aml_package_path = '/'.join(dirname(dirname(abspath(__file__))).split('/')[:-2]) + '/' + aml_package_name

    return aml_package_path


def get_abs_path(path):
    return abspath(path)


def save_data(data, filename, append_to_file = False, over_write_existing=False):

    if not over_write_existing:

        if exists(filename):
            warnings.warn("File exist by same name, renaming new file...")
            filename = filename[:-4]+time.strftime("_%b_%d_%Y_%H_%M_%S", time.localtime())+'.pkl'

    file_opt = 'wb' if not append_to_file else 'ab'
    
    output = open(filename, 'wb')

    # Pickle dictionary using protocol 0.
    pickle.dump(data, output)

    # Pickle the list using the highest protocol available.
    # pickle.dump(selfref_list, output, -1)

    output.close()


def load_data(filename):
	
    try:
        pkl_file = open(filename, 'rb')
    except Exception as e:
        raise e

    data = pickle.load(pkl_file)

    pkl_file.close()

    return data




