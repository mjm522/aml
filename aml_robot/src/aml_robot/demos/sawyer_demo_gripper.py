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


waypoints = [[  1.44433594e-03, -1.17865039e+00, -3.66015625e-03, 2.17480762e+00, 4.99707031e-03, 5.72320313e-01, 3.30896289e+00],
             [ 0.25643262,-0.5726875,-0.5376123,1.19443945,0.56056738,1.07112207,1.28155762],
             [ 0.37327246,0.09713184,-0.3957002,1.56641211,0.38141504,-1.4448964,2.12670801]]
#[ 0.17845996, -0.31037988,-0.31667383,1.19357324, 0.46605957,0.77225977, 1.36988184]

gopen_pos = 0.041667
gclosed_pos = 0.015

def goto_waypoints(wpts,limb):

    for wpt in wpts:
        limb.move_to_joint_pos(wpt)


limb = SawyerArm('right')


raw_input("WARNING: remove all obstacles from robot workspace. Insert any input to continue.")


limb.exec_gripper_cmd(gopen_pos)




rate = rospy.Rate(1) # 10hz

toggle_close = False
while not rospy.is_shutdown():


    print "CURRENT GRIPPER STATE:", limb.get_gripper_state()
    

    if toggle_close:
        limb.exec_gripper_cmd(gclosed_pos)

        rospy.sleep(1)

        goto_waypoints(reversed(waypoints),limb)
    else:

        goto_waypoints(waypoints,limb)

        limb.exec_gripper_cmd(gopen_pos)

    c = raw_input("Input close or open (c/o):")


    if c == "c":
        toggle_close = True
    if c == "o":
        toggle_close = False



    # rate.sleep()

    
    

