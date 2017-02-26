import aml_data_collec_utils

import cv2
import rospy
import numpy as np

from aml_data_collec_utils.core.sample import Sample
from aml_data_collec_utils.core.data_manager import DataManager
from aml_data_collec_utils.core.data_recorder import DataRecorder

from aml_data_collec_utils.config import config
from aml_robot.baxter_robot import BaxterArm

rospy.init_node('test_record_sample', anonymous=True)


class DummyTaskInterface(object):


    def __init__(self):
        pass

    def get_status(self):

        return {"task_status": "dummy_status"}
    def get_effect(self):

        return {"task_state": "dummy_effect"}



task = DummyTaskInterface()
robot = BaxterArm('left')
sample_recorder = DataRecorder(robot_interface=robot, 
                                           task_interface=task,
                                           data_folder_path=config['data_folder_path'],
                                           data_name_prefix='test_push_data',
                                           num_samples_per_file=5)


# dm = DataManager(append_to_last_file=True, 
#                  data_folder_path=config['data_folder_path'], 
#                  data_name_prefix='test_push_data',
#                  num_samples_per_file=5)



rospy.sleep(5)
loop_rate = rospy.Rate(30)

# def on_shutdown():
#     sample_recorder.save_data_now()
#     print "Saving!"

# rospy.on_shutdown(on_shutdown)



sample = Sample()



terminal = False

count = 0

start_time = rospy.Time.now()

sample_recorder.start_record({"dummy_action": "pumba!"})

while not rospy.is_shutdown():

    #sample_recorder.record_once({"dummy_action"},terminal=terminal)
    #terminal = not terminal


    time_elapsed = rospy.Time.now() - start_time


    if time_elapsed >= rospy.Duration(5):
        print "time to stop"
        

        sample_recorder.stop_record(task_status=True)

        start_time = rospy.Time.now()

        sample_recorder.start_record({"dummy_action": "pumba!"})


    # sample_recorder.record_once(None,task_status=True)



    # data = {"robot_state": np.random.rand(1), "box_state": np.random.rand(1), "action": np.random.rand(2)}

    # sample.add(data)
    # print "Another data point", sample._id, sample.size()

    # if sample.size() > 2:
    #   print "creating new sample"

    #   dm.add(sample)

    #   sample = Sample()

    loop_rate.sleep()




print "BYE!"


