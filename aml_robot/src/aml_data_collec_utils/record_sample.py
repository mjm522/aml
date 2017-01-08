import numpy as np
import cPickle
from os.path import dirname, abspath
import rospy

class Sample():

    def __init__(self, sample_id):

        self._data = {}

        self._data['sample_id'] = sample_id

        self._data['state'] = []

        self._data['task_action'] = []

        self._sample._data['task_status'] = False

        #this could be set from a hyper param file
        data_folder_path = dirname(dirname(abspath(__file__))) + '/data/'

        self._data_file = data_folder_path + ('push_data_sample_%02d.pkl' % sample_id)

def write_sample(self):

    with open(self._data_file, 'wb') as data_file:
        cPickle.dump(data_file, self._data)


class RecordSample():

    def __init__(self, robot_interface, record_rate=30):

        self._robot = robot_interface
        
        self._stale_data = None
        
        self._record = False

        self._record_period  = rospy.Duration(1.0/record_rate)

        self._sample_idx = None

    def configure(self, task_action):

        if self._samle_idx is None:

            self._samle_idx = 0
        else:
            self._samle_idx += 1
        
        self._callback = rospy.Timer(self._record_period, self.record_sample(task_action=task_action))

    def start_record(self, action):
        
        self._record = True
        
        self._configure(action)

    def stop_record(self, task_status):
        
        self._record = False
        
        self._callback.shutdown()

        #update the task status was a success and fail
        self._sample._data['task_status'] = task_status

        self._sample.write_sample()

    def check_sample(self, time_stamp):

        if self._old_time_stamp is None:

            self._old_time_stamp = time_stamp


        if time_stamp >= self._old_time_stamp + self._record_interval:

            return True

        else:

            return False

    def record_sample(self, task_action):

        if self._record:

            self._sample = Sample(self._sample_idx)

            data = self._robot._state

        if self.check_sample(data['timestamp']):

            self._sample._data['state'].append(data)
            self._sample._data['task_action'].append(task_action)

            

