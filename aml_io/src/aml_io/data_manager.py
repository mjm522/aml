import matplotlib.image as mpimg

from aml_io.io_tools import load_data, save_data

import numpy as np
import pickle

class DataManager(object):

    def __init__(self, data = []):

        self._data = data

        self._next_sample_id = 0

    @classmethod
    def from_file(cls, filepath, filenames):

        data = []

        for filename in filenames:

            #joining all filenames
            data += load_data(filepath+filename)

        return cls(data)


    def save(self, filename):

        output = open(filename, 'wb')


        pickle.dump(self._data, output)


        output.close()


    def add(self, sample):

        sample['sample_id'] = self._next_sample_id
        self._next_sample_id += 1
        
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

    def pack_data(self, keys, sub_keys=None, ids=None):

        #keys can be the key inside each data points, while sub_keys are keys inside each key
        #important if a key in keys does not have a corresponding subkey, but other keys have a subkey,
        #then you need to call it in this way
        #keys=[key_1, key_2], sub_key_1 = [sub_key_1_1, sub_key_1_2], there are no subkeys for key_2
        #subkeys=[sub_key_1, [None]]

        data = []
        
        data_list = self.select_data(ids)

        num_keys  =  len(keys)

        for datum in data_list:
            
            #a data point inside data
            point = []

            for i in range(num_keys):

                no_sub_keys = False

                if sub_keys is not None:

                    for sub_key in sub_keys[i]:

                        if sub_key is not None:

                            point = np.r_[point, datum[keys[i]][sub_key]]

                        else:

                            no_sub_keys = True

                            break

                else:

                    no_sub_keys = True

            
            if no_sub_keys:

                point = np._r_[point, datum[keys[i]]]

            data.append(point)

        return data


    def pack_sample(self, x_keys, x_sub_keys, y_keys, y_sub_keys, ids=None):

        if ids[0]=='all':

            ids = range(len(self._data))

        data_x = self.pack_data(keys=x_keys, sub_keys=x_sub_keys, ids=ids)
        
        data_y = self.pack_data(keys=y_keys, sub_keys=y_sub_keys, ids=ids)

        return data_x, data_y


    def create_sample(self):

        return {'filled': False, 'stale': False}


    def size(self):
        return len(self._data)
















