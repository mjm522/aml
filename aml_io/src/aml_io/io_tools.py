import time
import warnings

# For compatibility with python 2.7 and 3
try:
    import cPickle as pickle
except ImportError:
    import pickle

from os.path import exists, join, dirname, abspath
import os
from aml_io.log_utils import aml_logging

def get_aml_package_path(aml_package_name=None):

    #if the package name is none, it returns path to aml folder
    if aml_package_name is None:

        aml_package_path = '/'.join(dirname(dirname(abspath(__file__))).split('/')[:-2])

    else:

        aml_package_path = '/'.join(dirname(dirname(abspath(__file__))).split('/')[:-2]) + '/' + aml_package_name

    return aml_package_path


def get_abs_path(path):
    return abspath(path)


def crawl(path, extension_filter='urdf'):

    crawled_paths = {}
    for dirpath, subs, files in os.walk(path):
        for file in files:

            if extension_filter is None:
                crawled_paths[file] = os.path.join(dirpath, file)
            elif file.split('.')[-1] == extension_filter:
                crawled_paths[file] = os.path.join(dirpath, file)


    return crawled_paths

def get_file_path(file, search_paths):

    if not isinstance(search_paths,list):
        search_paths = [search_paths]



    crawled_paths = {}
    ext = file.split('.')[-1]

    for path in search_paths:
        crawled_paths.update(crawl(path,ext))


    file_path = crawled_paths.get(file,None)

    if file_path is None:
        aml_logging.warning('File %s not found, returning None path.'%(file,))

    return file_path




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




