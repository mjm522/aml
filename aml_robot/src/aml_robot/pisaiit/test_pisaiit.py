#!/usr/bin/env python

import rospy

from aml_robot.pisaiit.pisaiit_robot import PisaIITHand

from functools import partial

import numpy as np
import quaternion


def callback(agent, msg):
    pass


# print(agent.c)
# print("Hello!")


class SomeObj:
    def __init__(self):
        self.c = 0


rospy.init_node('baxter_test', anonymous=True)

obj = SomeObj()

robot = PisaIITHand('right', partial(callback, obj))
start_pos, start_ori = robot.ee_pose()

goal_pos = np.array([0.95, -0.08, -0.11])
goal_ori = quaternion.as_float_array(start_ori)
print "GOALORI: ", goal_ori

rate = rospy.Rate(5)  # 10hz
while not rospy.is_shutdown():
    obj.c += 1
    print(robot.inverse_kinematics(start_pos, goal_ori))
    rate.sleep()
