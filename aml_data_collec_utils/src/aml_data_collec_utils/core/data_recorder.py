
import rospy

from functools import partial

from aml_data_collec_utils.core.sample import Sample
from aml_data_collec_utils.core.data_manager import DataManager
from aml_io.log_utils import aml_logging

from aml_io.convert_tools import image2string, dimage2string

import numpy as np
import quaternion
import copy


class DataRecorder(object):

    def __init__(self, robot_interface, task_interface, data_folder_path=None,  
                       data_name_prefix=None, num_samples_per_file=1000, record_rate = 20):



        self._robot            = robot_interface

        self._task             = task_interface

        self._old_time_stamp   = None

        self._data_name_prefix = data_name_prefix+'check_depth_data'

        self._data_folder_path = data_folder_path

        self._record_period = rospy.Duration(1.0/record_rate)

        self._data_man = DataManager(append_to_last_file=True, 
                                     data_folder_path=data_folder_path, 
                                     data_name_prefix=data_name_prefix,
                                     num_samples_per_file=num_samples_per_file)

        self._sample_idx  = self._data_man._last_sample_idx + 1

        self._sample      = Sample()

        self._callback = None
            

    #this piece of funtion is useful only in case of continous recording
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

    def record_once(self, task_action, terminal = False):

        robot_state = copy.deepcopy(self._robot._state)
        task_state = self._task.get_effect()


        

        if not self.check_sample(robot_state['timestamp']):
            aml_logging.warning("DataRecorder.record_once: check_sample failed")
            return 

        #np.quaternion causes problem, hence convert to array
        if isinstance(robot_state['ee_ori'], np.quaternion):
            robot_state['ee_ori'] = quaternion.as_float_array(robot_state['ee_ori'])[0]
        
        #compressing image
        if robot_state['rgb_image'] is not None:
            robot_state['rgb_image'] = image2string(image_in=robot_state['rgb_image'],fmt="jpeg")
            # dimage2string(image_in=robot_state['depth_image'])
            robot_state['depth_image'] = None

        data = {}
        data.update(robot_state)
        data.update(task_state)
        data.update({'task_action': task_action, 'terminal' : terminal})

        self._sample.add(data)

        # aml_logging.info("Sample size: %d"%(self._sample.size,))

        if terminal:
            self.save_sample()


    #in case keyboard interrupt is called, then execute this!
    def save_data_now(self):

        self._data_man.write_data()



    def record_sample(self, task_action, event):

        if self._record:

            self.record_once(task_action)

    def start_record(self, task_action):

        self._callback = rospy.Timer(self._record_period, partial(self.record_sample,task_action))

        self._record = True


    def save_sample(self):

        self._data_man.add(self._sample)

        self._sample = Sample()


    def stop_record(self, task_status):
        
        self._record = False
        
        if self._callback is None:
            
            print "Nothing to kill, since the recorder was not started ..."
        
        else:
            
            self._callback.shutdown()

            #update the task status was a success and fail

            self._sample.set_valid(task_status)

            # Setting last data point as terminal


            if self._sample.size > 0:
                self._sample.set(-1,'terminal', True)
                self.save_sample()
            else:
                self._sample = Sample()