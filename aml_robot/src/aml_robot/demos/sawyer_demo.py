#!/usr/bin/env python

# A simply python controller for Baxter compatible with GPS (TODO)


import rospy

import intera_interface

from aml_robot.sawyer_robot import SawyerArm

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

rospy.init_node('sawyer_test', anonymous=True)

obj = SomeObj()

limb = SawyerArm('right',partial(callback,obj))
limb.untuck_arm()

rospy.sleep(3)

start_pos, start_ori = limb.get_ee_pose()

goal_pos = start_pos + np.array([0.2,-0.08,-0.11])
goal_ori = quaternion.as_float_array(start_ori)
print "GOALORI: ", goal_ori



rate = rospy.Rate(10) # 10hz
while not rospy.is_shutdown():
	obj.c += 1

	success, joints = limb.ik(goal_pos,goal_ori)

	print success, joints

	print "CURRENT STATE:", limb.angles()
	limb.exec_position_cmd(joints)

	rate.sleep()

	
	

