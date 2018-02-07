#!/usr/bin/env python

# A simple python controller for Baxter compatible with GPS (TODO)


import rospy

import baxter_interface
import baxter_external_devices

from aml_robot import baxter_robot

from std_msgs.msg import (
    UInt16,
)

from baxter_interface import CHECK_VERSION

from aml_perception import camera_sensor
from functools import partial


import numpy as np
import quaternion

# This is a callback that can be registered in the constructor of BaxterArm
# This is called periodically, and msg is the state of the Baxter
def callback(state_msg):
    print state_msg['ee_point']
    # print("Hello!")
incr = np.array([0,0,0])
rospy.init_node('baxter_test', anonymous=True)

_rs = baxter_interface.RobotEnable(CHECK_VERSION)
_rs.enable()

# limb = baxter_robot.BaxterArm('right',callback)
limb = baxter_robot.BaxterArm('right')
limb.untuck_arm()


start_pos, start_ori = limb.get_ee_pose()

goal_ori = quaternion.as_float_array(start_ori)
incr = np.array([0.005,-0.05,-0.05])

rate = rospy.Rate(5) # 10hz
while not rospy.is_shutdown():
    goal_pos = start_pos + incr
    # print "GOALORI: ", goal_ori
    print "GOAL POSITION:", goal_pos
    current_ee_pos, current_ee_ori = limb.get_ee_pose()
    print "ACTUAL POSITION:", current_ee_pos

    success, goal_joint_angles = limb.ik(goal_pos,goal_ori)
    if not success:
        incr = - incr
    # print success, goal_angles
    


    limb.exec_position_cmd(goal_joint_angles)
    start_pos = goal_pos
    
    rate.sleep()

    
    

