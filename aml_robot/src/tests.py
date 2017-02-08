#!/usr/bin/env python

# A simply python controller for Baxter compatible with GPS (TODO)


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

def callback(agent,msg):
	pass
	# print(agent.c)
	# print("Hello!")


class SomeObj:
	def __init__(self):
		self.c = 0

rospy.init_node('baxter_test', anonymous=True)

_rs = baxter_interface.RobotEnable(CHECK_VERSION)
_rs.enable()


obj = SomeObj()

limb = baxter_robot.BaxterArm('right',partial(callback,obj))
start_pos, start_ori = limb.get_ee_pose()

goal_pos = np.array([0.95,-0.08,-0.11])
goal_ori = quaternion.as_float_array(start_ori)[0]
print "GOALORI: ", goal_ori

rate = rospy.Rate(5) # 10hz
while not rospy.is_shutdown():
	obj.c += 1
	print(limb.ik(start_pos))
	rate.sleep()

	
	

