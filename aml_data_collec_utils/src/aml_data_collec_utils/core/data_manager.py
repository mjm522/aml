import matplotlib.image as mpimg

from aml_io.io_tools import load_data, save_data
from aml_io.log_utils import aml_logging
from aml_data_collec_utils.config import config


from aml_data_collec_utils.core.sample import Sample

import os

import numpy as np
import pickle


class DataManager(object):

    def __init__(self, append_to_last_file=True, data_folder_path=None, data_name_prefix=None, num_samples_per_file=1000):

        
        self._last_sample_idx = 0 

        #this could be set from a hyper param file
        if data_folder_path is None:
            
            self._data_folder_path = config['data_folder_path']

        else:
            
            self._data_folder_path = data_folder_path

        #maximum number of samples that is allowed per file
        self._num_samples_per_file = num_samples_per_file

        if data_name_prefix is None:

            self._data_name_prefix = 'data'

        else:

            self._data_name_prefix = data_name_prefix


        create_new_data = False

        data_names = []

        #read files in the folder that starts with self._sample_name_prefix name
        for file in os.listdir(self._data_folder_path):

            if file.startswith(self._data_name_prefix):
            
                data_names.append(file)

        #if append to last file is true and if the data_names are non-empty then add to existing sample,
        #else create a new data sample!
        if append_to_last_file and data_names:

            self._data_idx = len(data_names)

            #this is in case if the current data is empty
            #so this shouldn't be the case when data_idx is 1
            print max(1, self._data_idx -1)

            _prev_data      = self.read_data(max(1, self._data_idx -1))

            self._data      = self.read_data(self._data_idx)

            #this case is to check in case you have given keyboard interrupt in the middle of recording
            #and you have an incomplete sample, so we will replace it.

            if self._data:
                
                if  self.check_sample(self._data[-1]):

                    self._last_sample_idx = self._data[-1].get_id()
                
                else:

                    del self._data[-1]

                    self._last_sample_idx = self._data[-1].get_id()


                Sample.configure_id_counter(self._last_sample_idx+1)


                if self._last_sample_idx >= self._num_samples_per_file:
                    print "WARNING: Max number of samples per file in place, saving as a new sample..."
                    create_new_data = True
            else:

                #so this is the case when the current data is empty, 
                #but we need to check if previoud data exist as well.
                if _prev_data:

                    self._last_sample_idx = _prev_data[-1].get_id()

                else:
                    create_new_data = True
                    self._data_idx  = 0



        else:
            create_new_data = True
            self._data_idx  = 0

        if create_new_data:

            self.create_new_data()

    def create_new_data(self):
    
        self._data_idx += 1
        self._data     = []

    def write_data(self):

        data_file = self._data_folder_path + self._data_name_prefix + ('_%02d.pkl' % self._data_idx)

        save_data(self._data, data_file)

        self.create_new_data()


    def read_data(self, data_idx):

        data_file = self._data_folder_path + self._data_name_prefix +  ('_%02d.pkl' % data_idx)

        if not os.access(data_file, os.R_OK):
                
            aml_logging.info("Cannot read file at '%s'" % (data_file))

            return None

        else:

            data = None
            try:
                data = load_data(data_file)

                self._data = data
            except Exception as e:
                print "Unable to load data, file corrupted, creating new data file"
                os.remove(data_file)
                self._data_idx -= 1
                self.create_new_data()

            return data


    def add(self, sample):

        saved_data = False

        #if more number of samples came in, then make a new sample, else append to existing sample
        if len(self._data) >= self._num_samples_per_file:

            print "Saving the data file ..."

            self.write_data()

            #this flag is so that it doen't increment sample_id if the data is saved.
            saved_data = True

        else:

            print "New sample added with size ", sample.size()
        
            self._data.append(sample)

        return saved_data


    #this piece of funtion is useful only in case of continous recording
    def check_sample(self, sample):


        return sample.is_valid()
        


    def get_sample(self, idx, key):
        assert( idx >= 0 and idx < len(self._data) )

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
                
                x = np.r_[x,datum[key]['position'],datum[key]['angle']]
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
















