#!/usr/bin/env python

# A simply python controller for Baxter compatible with GPS (TODO)


import rospy

import intera_interface
import intera_external_devices

from aml_robot.sawyer_robot import SawyerArm

from aml_perception import camera_sensor
from functools import partial


import numpy as np
import quaternion


rospy.init_node('sawyer_test', anonymous=True)

gopen_pos = 0.041667
gclosed_pos = 0.015



limb = SawyerArm('right')


raw_input("WARNING: remove all obstacles from robot workspace. Insert any input to continue.")


limb.exec_gripper_cmd(gopen_pos)




rate = rospy.Rate(1) # 10hz

toggle_close = False
while not rospy.is_shutdown():


    print "CURRENT STATE:", limb.angles()

    joints = limb.angles()
    

    if toggle_close:
        joints[7] = gclosed_pos
        limb.exec_position_cmd(joints)

        rospy.sleep(1)
    else:
        rospy.sleep(1)

        joints[7] = gopen_pos
        limb.exec_position_cmd(joints)

    c = raw_input("Input close or open (c/o):")


    if c == "c":
        toggle_close = True
    if c == "o":
        toggle_close = False



    # rate.sleep()

    
    

