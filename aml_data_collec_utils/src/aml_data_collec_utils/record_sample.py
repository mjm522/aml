import os
import rospy
import numpy as np
import quaternion

from aml_data_collec_utils.config import config

from functools import partial
from aml_io.io_tools import save_data, load_data


class DataManager():

    def __init__(self, append_to_last_file=True, data_folder_path=None, data_name_prefix=None, num_samples_per_file=1000):

    
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

                    self._last_sample_idx = self._data[-1]['sample_id']
                
                else:

                    del self._data[-1]

                    self._last_sample_idx = self._data[-1]['sample_id']

                if self._last_sample_idx >= self._num_samples_per_file:
                    print "WARNING: Max number of samples per file in place, saving as a new sample..."
                    create_new_data = True
            else:

                #so this is the case when the current data is empty, 
                #but we need to check if previoud data exist as well.
                if _prev_data:

                    self._last_sample_idx = _prev_data[-1]['sample_id']

                else:
                    create_new_data = True
                    self._data_idx  = 0


        else:
            create_new_data = True
            self._data_idx  = 0

        if create_new_data:

            self.create_new_data()


    def check_sample(self, sample):

        bad_sample =  (sample['state_before'] is None or sample['state_after'] is None or sample['task_action'] is None\
            or sample['task_before'] is None or sample['task_after'] is None or sample['task_status'] is None)

        return not bad_sample
            
    def create_new_data(self):
    
        self._data_idx += 1
        self._data     = []
        self._last_sample_idx = 0


    def append_data(self, sample):

        saved_data = False

        #if more number of samples came in, then make a new sample, else append to existing sample
        if len(self._data) >= self._num_samples_per_file:

            print "Saving the data file ..."

            self.write_data()

            #this flag is so that it doen't increment sample_id if the data is saved.
            saved_data = True

        else:

            print "New sample (no:%d) added..."%(sample._contents['sample_id'],)
        
            self._data.append(sample._contents)

        return saved_data

    def write_data(self):

        data_file = self._data_folder_path + self._data_name_prefix + ('_%02d.pkl' % self._data_idx)

        save_data(self._data, data_file)

        self.create_new_data()


    def read_data(self, data_idx):

        data_file = self._data_folder_path + self._data_name_prefix +  ('_%02d.pkl' % data_idx)

        if not os.access(data_file, os.R_OK):
                
            rospy.loginfo("Cannot read file at '%s'" % (data_file))

            return None

        else:

            try:
                data = load_data(data_file)
            except Exception as e:
                print "Unable to load data, file corrupted, creating new data file"
                os.remove(data_file)
                self._data_idx -= 1
                self.create_new_data()

            return 

    def get_specific_sample(self, data_idx, sample_idx):

        return read_data(data_idx)[sample_idx]


class Sample():

    def __init__(self, sample_id):

        self._contents                = {}

        self._contents['sample_id']    = sample_id

        self._contents['state_before'] = None

        self._contents['state_after']  = None

        self._contents['task_action']  = None

        self._contents['task_before']  = None

        self._contents['task_after']   = None

        self._contents['task_status']  = None


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

        self._sample_idx  = self._data_man._last_sample_idx + 1

        self._sample      = Sample(self._sample_idx)

        self._callback = None

    

    def save_sample(self):

        #this check is prevent increment of sample_idx if we are starting a fresh data file
        if not self._data_man.append_data(self._sample):

            self._sample_idx += 1

        self._sample      = Sample(self._sample_idx)
            

    #this piece of funtion is useful only in case of continous recording
    def check_sample(self, sample):

        return True

    def record_once(self, task_action, task_status=False):

        robot_state = self._robot._state
        task_state  = self._task.get_effect()

        #np.quaternion causes problem, hence convert to array

        if isinstance(robot_state['ee_ori'], np.quaternion):

            robot_state['ee_ori'] = quaternion.as_float_array(robot_state['ee_ori'])[0]


        #if task action is none, that means there is no task execution.
        if task_action is None:

            #all the stuff after the push is being done.

            self._sample._contents['state_after'] = robot_state

            self._sample._contents['task_after'] = task_state

            self._sample._contents['task_status'] = task_status

            self.save_sample()

        else:

            #all the stuff before a push is being done.

            self._sample._contents['task_action'] = task_action

            self._sample._contents['state_before'] = robot_state
            
            self._sample._contents['task_before'] = task_state

    #in case keyboard interrupt is called, then execute this!
    def save_data_now(self):

        self._data_man.write_data()



    def record_sample(self, task_action, event):

        if self._record:

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

    def start_record(self, task_action):
        
        self._record = True

        self._callback = rospy.Timer(self._record_period, partial(self.record_sample,task_action))

    def stop_record(self, task_status):
        
        self._record = False
        
        if self._callback is None:
            
            print "Nothing to kill, since the recorder was not started ..."
        
        else:
            
            self._callback.shutdown()

            #update the task status was a success and fail
            self._sample._data['task_status'] = task_status

            self._sample.write_sample()
