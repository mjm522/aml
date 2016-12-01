#!/usr/bin/env python

# A simply python controller for Baxter compatible with GPS (TODO)


import rospy

import baxter_interface
import baxter_external_devices

from std_msgs.msg import (
    UInt16,
)

from baxter_interface import CHECK_VERSION

import baxter_robot

from functools import partial

#from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES

import numpy as np

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

left_limb = baxter_robot.BaxterArm('left',partial(callback,obj))

rate = rospy.Rate(10) # 10hz
while not rospy.is_shutdown():
	obj.c += 1
	rate.sleep()
	

