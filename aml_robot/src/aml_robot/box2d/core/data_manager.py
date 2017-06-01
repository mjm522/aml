#TODO: REMOVE THIS FILE, THERE IS SAME FILE IN AML_IO

import os
import matplotlib.image as mpimg
from aml_io.io_tools import load_data, save_data

import numpy as np
import pickle

class DataManager(object):

    def __init__(self, data = [], samples_per_file=5, file_prefix='box2d_push_data', data_folder='./data'):

        self._data = data

        self._next_sample_id = 0
        self._file_idx = 0
        self._file_prefix = file_prefix
        self._data_folder = data_folder

        if not os.path.exists(self._data_folder):
            os.makedirs(self._data_folder)

        self._num_samples_per_file = samples_per_file

    @classmethod
    def from_file(cls, filename, data_folder='.'):

        return cls(data=load_data(data_folder + '/' + filename))


    def save(self, filename):

        output = open(filename, 'wb')


        pickle.dump(self._data, output)


        output.close()


    def add(self, sample, seperate_files=False):

        sample['sample_id'] = self._next_sample_id
        self._next_sample_id += 1
        
        if seperate_files:
            if (self._next_sample_id > 0) and (self._next_sample_id % self._num_samples_per_file == 0):
                filename = self._data_folder + '/' + self._file_prefix + '_' + str(self._file_idx) + '.pkl'
                self.save(filename)
                self._file_idx += 1
                self._data = []
            else:
                self._data.append(sample)
        else:
            self._data.append(sample)

        
    def get_sample(self, idx, key):
        assert( idx < len(self._data) )

        return self._data[idx][key]


    def get_last(self):
        if self.size() > 0:
            return self._data[-1]
        else:
            return None

    def get_last_elems(self, n = 1):

        if self.size() > 0:
            return self._data[-(n+1):-1]
        else:
            return None

    def get_last_elems(self, n = 1):
        
        if self.size() > 0:
            return self._data[-(n+1):-1]
        else:
            return None

    def get_last_ids(self, n = 1):
        
        return range(self.size()-n,self.size())


    def select_data(self, ids = None):

        data = []
        if ids == None:
            data = self._data
        else:
            data = [self._data[idx] for idx in ids]

        return data

    def pack_data_x(self, keys, ids = None):

        data_x = []
        
        data = self.select_data(ids)

        for datum in data:
            x = []
            for key in keys:
                
                x = np.r_[x,np.multiply(datum[key]['position']- np.array([16.0,12.0]),[1./32.0,1./24.0]),datum[key]['angle']]
                    # datum[key]['position'], .., datum[key]['angle'],datum[key]['angular_velocity'], datum[key]['linear_velocity']

            data_x.append(x)


        return data_x

    def pack_data_y(self, ids = None):

        data_y = []

        data = self.select_data(ids)

        for datum in data:

            data_y.append([datum['push_action'][0][4]])


        return data_y


    def pack_data(self, keys, ids = None):

        data_x = self.pack_data_x(keys, ids)
        data_y = self.pack_data_y(ids)


        return data_x, data_y


    def create_sample(self):

        return {'filled': False, 'stale': False}


    def size(self):
        return len(self._data)
















