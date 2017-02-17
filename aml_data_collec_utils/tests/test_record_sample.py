import aml_data_collec_utils

import cv2
import rospy

from aml_data_collec_utils.record_sample import RecordSample
from aml_robot.baxter_robot import BaxterArm

rospy.init_node('test_record_sample')


class DummyTaskInterface(object):


    def __init__(self):
        pass

    def get_status(self):

        return "dummy_status"
    def get_effect(self):

        return "dummy_effect"



task = DummyTaskInterface()
robot = BaxterArm('left')
sample_recorder = RecordSample(robot_interface=robot, 
                                           task_interface=task,
                                           data_folder_path=None,
                                           data_name_prefix='test_push_data',
                                           num_samples_per_file=5)

rospy.sleep(10)
loop_rate = rospy.Rate(1)

def on_shutdown():
    sample_recorder.save_data_now()
    print "Saving!"

rospy.on_shutdown(on_shutdown)

while not rospy.is_shutdown():


    sample_recorder.record_once({"dummy_sample"},task_status=True)
    sample_recorder.record_once(None,task_status=True)


    print "Another sample"


    loop_rate.sleep()







# data = data_man.read_data(1)

# print type(data)

# for sample in data:

#     print "##########################################################################################################"

#     keys = sample.keys()

#     for key in sample:

#         print "****************************************************************************************************"
#         print "Key \t", key
#         print "Value \t", sample[key]
#         print "****************************************************************************************************"

# quit = False

# key = 0

# data = data_man.read_data(5)

# sample_idx = 0

# while not quit:

#     sample = data[sample_idx]


#     image = sample['state_before']['rgb_image']
#     cv2.imshow("RGB Image Before", image)

#     image = sample['state_after']['rgb_image']
#     cv2.imshow("RGB Image After", image)

#     print "SAMPLE_ID: \t", sample['sample_id']

#     print "STATUS: \t", sample['task_status']

#     print "Start location of the box \n", sample['task_before']

#     print "Push action \n", sample['task_action']

#     print "End location of the box \n", sample['task_after']

#     key = cv2.waitKey(0)

#     print key
#     if key == 65363:
#     	sample_idx += 1
#     elif key == 65361:
#     	sample_idx = max(0,sample_idx-1)

#     if key == 27 or sample_idx >= len(data):
#         quit = True



print "BYE!"


