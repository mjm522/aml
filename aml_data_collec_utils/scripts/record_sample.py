import os
import rospy
import numpy as np
import quaternion
from os.path import dirname, abspath

from functools import partial
from aml_io.io_tools import save_data, load_data


class DataManager():

    def __init__(self, append_to_last_file=True, data_folder_path=None, data_name_prefix=None, num_samples_per_file=1000):

        #this could be set from a hyper param file
        if data_folder_path is None:
            
            self._data_folder_path = dirname(dirname(abspath(__file__))) + '/data/'

        else:
            
            self._data_folder_path = data_folder_path

        if data_name_prefix is None:

            self._data_name_prefix = 'data'

        else:

            self._data_name_prefix = data_name_prefix


        create_new_data = False

        data_names = []

        #read files in the folder that starts with self._sample_name_prefix name
        if any(file.startswith(self._data_name_prefix) for file in os.listdir(self._data_folder_path)):
            
            data_names.append(file)

        #if append to last file is true and if the data_names are non-empty then add to existing sample,
        #else create a new data sample!
        if append_to_last_file and data_names:

            self._data_idx = len(data_names)+1

            self._data     = self.read_data(self._data_idx)

            #this case is to check in case you have given keyboard interrupt in the middle of recording
            #and you have an incomplete sample, so we will replace it.
            if self.check_sample(self._data[-1]):

                self._last_sample_idx = self._data[-1]['sample_id']
            
            else:

                self._last_sample_idx = self._data[-1]['sample_id'] - 1

            if self._last_sample_idx >= self._num_samples_per_file:
                print "WARNING: Max number of samples per file in place, saving as a new sample..."
                create_new_data = True
        else:
            create_new_data = True

        if create_new_data:

            self.create_new_data()


    def check_sample(self, sample):

        bad_sample =  (sample['state_before'] is None or sample['state_after'] is None or sample['task_action'] is None/
            sample['task_before'] is None or sample['task_after'] is None or sample['task_status'] is None)

        return not bad_sample

            
    def create_new_data(self):

        self._data_idx = len(data_names) + 1
        self._data     = []
        self._last_sample_idx = 1


    def append_data(self, sample):

        #if more number of samples came in, then make a new sample, else append to existing sample
        if sample['sample_id'] >= self._num_samples_per_file:

            self.write_data()

            self.create_new_data()

            self._data.append(sample)

        else:
        
            self._data.append(sample)

    def write_data(self):

        data_file = self._data_folder_path + self._data_name_prefix + ('_data_%02d.pkl' % self._data_idx)

        save_data(data_file)

        self._data_idx += 1


    def read_data(self, data_idx):

        data_file = self._data_folder_path + self._data_name_prefix +  ('_data_%02d.pkl' % data_idx)

        if not os.access(data_file, os.R_OK):
                
            rospy.logerr("Cannot read file at '%s'" % (args.file,))

            return None

        else:

            return load_data(data_file)

    def get_specific_sample(self, data_idx, sample_idx):

        return read_data(data_idx)[sample_idx]


class Sample():

    def __init__(self, sample_id):

        self._sample                 = {}

        self._sample['sample_id']    = sample_id

        self._sample['state_before'] = None

        self._sample['state_after']  = None

        self._sample['task_action']  = None

        self._sample['task_before']  = None

        self._sample['task_after']   = None

        self._sample['task_status']  = None


class RecordSample():

    def __init__(self, robot_interface, task_interface, data_folder_path=None,  
                       data_name_prefix=None, num_samples_per_file=1000):

        self._robot            = robot_interface

        self._task             = task_interface

        self._old_time_stamp   = None

        self._data_name_prefix = data_name_prefix

        self._data_folder_path = data_folder_path

        self._data_man = DataManager(append_to_last_file=True, 
                                     data_folder_path=data_folder_path, 
                                     data_name_prefix=data_name_prefix,
                                     num_samples_per_file=num_samples_per_file)

        self._sample_idx     = self._data_man._last_sample_idx + 1

        self._sample    = Sample(self._sample_idx)


    def save_sample(self):

        self._data_man.append_data(self._sample)

        self._sample_idx += 1
            

    def check_sample(self, time_stamp):

        if self._old_time_stamp is None:

            self._old_time_stamp = time_stamp

        #a small function to compute time from timestamps
        def time_compute(time_stamp):

            return time_stamp['secs']  + 1e-9*time_stamp['nsecs']
        
        if time_compute(time_stamp) >= time_compute(self._old_time_stamp):

            return True

        else:

            return False

    def record_once(self, task_action=None, task_status=False):

        robot_state = self._robot._state
        task_state  = self._task.get_effect()

        if self.check_sample(data['timestamp']):

            #np.quaternion causes problem, hence convert to array
            if isinstance(robot_state['ee_ori'], np.quaternion):

                robot_state['ee_ori'] = quaternion.as_float_array(robot_state['ee_ori'])[0]


            #if task action is none, that means there is no task execution.
            if task_action is None:

                self._sample['state_after'] = robot_state

                self._sample['task_after'] = task_state

                self._sample['task_status'] = task_status

                self.save_sample():

            else:

                self._sample['task_action'] = task_action

                self._sample['state_before'] = robot_state
                
                self._sample['task_before'] = task_state

    def save_data_now(self):

        self._data_man.write_data()
