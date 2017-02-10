import os
import rospy
import numpy as np
import quaternion
from os.path import dirname, abspath

from functools import partial
from aml_io.io_tools import save_data, load_data

class Sample():

    def __init__(self, sample_id=None, data_folder_path=None, sample_name_prefix=None):

        self._data                = {}

        self._sample_id           = sample_id

        self._data['sample_id']   = sample_id

        if sampling_rate is None:

            print "WARNING: Sampling rate not stored, pass it if you want to store it."

        self._data['state']       = []

        self._data['task_action'] = []

        self._data['task_effect'] = []

        self._data['task_status'] = False

        #this could be set from a hyper param file
        if data_folder_path is None:
            
            self._data_folder_path = dirname(dirname(abspath(__file__))) + '/data/'

        else:
            
            self._data_folder_path = data_folder_path

        if sample_name_prefix is None:

            self._sample_name_prefix = 'unknown_task_name'

        else:

            self._sample_name_prefix = sample_name_prefix

    def write_sample(self):

        data_file = self._data_folder_path + self._sample_name_prefix + ('_sample_%02d.pkl' % self._sample_id)

        save_data(data=self._data, filename=data_file, append_to_file = True)

    def get_sample(self, sample_id):

        data_file = self._data_folder_path + self._sample_name_prefix +  ('_sample_%02d.pkl' % sample_id)

        data = load_data(data_file)

        return data


class RecordSample():

    def __init__(self, robot_interface, task_interface, data_folder_path=None,  
                       sample_name_prefix=None, num_samples_per_file=1000):

        self._robot          = robot_interface

        self._task           = task_interface

        self._old_time_stamp = None

        self._sample_name_prefix = sample_name_prefix

        self._data_folder_path = data_folder_path

        sample_names = []

        #read files in the folder that starts with self._sample_name_prefix name
        if any(file.startswith(self._sample_name_prefix) for file in os.listdir(self._data_folder_path)):
            
            sample_names.append(file)

        #check if list is empty
        if not sample_names:

            self._sample_idx     = 1

        else:

            #read the last file of the folder
            existing_sample = self._data_folder_path + sample_names[-1]

            if not os.access(existing_sample, os.R_OK):
                
                rospy.logerr("Cannot read file at '%s'" % (args.file,))
            
            else:

                #load the pkl file and get the id
                old_sample = load_data(existing_sample)

                #read the file, and take the sample_id of the last sample_id
                self._sample_idx = old_sample['sample_id'][-1]

        #this is the threshold, when 1000 
        self._num_samples_per_file = num_samples_per_file

    def configure(self):

        if self._sample_idx is None:

            self._sample_idx  = 1
            
        #configure the sample
        self._sample   = Sample(sample_id=self._sample_idx,
                                sampling_rate=self._record_rate,
                                data_folder_path=self._data_folder_path, 
                                sample_name_prefix=self._sample_name_prefix)


    def save_sample_ckpt(self, task_status):

        #update the task status was a success and fail
        self._sample._data['task_status'] = task_status

        #did we reach a limit?
        limit = (self._sample_idx%self._num_samples_per_file) == 0

        #did we reach limit or was the task_status = None ; meaning it was killed
        if  limit or task_status is None:
            
            self._sample.write_sample()

        else:

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

    def record_once(self, task_action):

        data = self._robot._state

        if self.check_sample(data['timestamp']):

            #np.quaternion causes problem, hence convert to array
            if isinstance(data['ee_ori'], np.quaternion):

                data['ee_ori'] = quaternion.as_float_array(data['ee_ori'])[0]

            else:

                print type(data['ee_ori'])

            self._sample._data['state'].append(data)
            self._sample._data['task_action'].append(task_action)
            self._sample._data['task_effect'].append(self._task.get_status())
