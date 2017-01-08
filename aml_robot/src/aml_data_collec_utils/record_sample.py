import rospy
import numpy as np
import quaternion
from os.path import dirname, abspath

from functools import partial
from aml_io.io import save_data, load_data

class Sample():

    def __init__(self, sample_id=None):

        self._data = {}

        self._sample_id = sample_id

        self._data['sample_id'] = sample_id

        self._data['state'] = []

        self._data['task_action'] = []

        self._data['task_status'] = False

        #this could be set from a hyper param file
        self._data_folder_path = dirname(dirname(abspath(__file__))) + '/aml_data_collec_utils/data/'

    def write_sample(self):

        data_file = self._data_folder_path + ('push_data_sample_%02d.pkl' % self._sample_id)

        save_data(data=self._data, filename=data_file)

    def get_sample(self, sample_id):

        data_file = self._data_folder_path + ('push_data_sample_%02d.pkl' % sample_id)

        data = load_data(data_file)

        return data


class RecordSample():

    def __init__(self, robot_interface, record_rate=30):

        self._robot          = robot_interface
        
        self._stale_data     = None
        
        self._record         = False

        self._record_period  = rospy.Duration(1.0/record_rate)

        self._sample_idx     = None

        self._old_time_stamp = None

    def configure(self, task_action):

        if self._sample_idx is None:

            self._sample_idx  = 0
        else:
            self._sample_idx += 1

        #configure the sample
        self._sample   = Sample(self._sample_idx)

        #create a sample record callback on an 
        #independent thread and record the data 
        self._callback = rospy.Timer(self._record_period, partial(self.record_sample,task_action))
    
    def start_record(self, task_action):
        
        self._record = True
        
        self.configure(task_action)

    def stop_record(self, task_status):
        
        self._record = False
        
        self._callback.shutdown()

        #update the task status was a success and fail
        self._sample._data['task_status'] = task_status

        self._sample.write_sample()

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

    def record_sample(self, task_action, event):

        if self._record:

            data = self._robot._state

        if self.check_sample(data['timestamp']):

            #np.quaternion causes problem, hence convert to array
            data['ee_ori'] = quaternion.as_float_array(data['ee_ori'])[0]

            self._sample._data['state'].append(data)
            self._sample._data['task_action'].append(task_action)

            

