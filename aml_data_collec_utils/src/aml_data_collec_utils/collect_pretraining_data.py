#!/usr/bin/env python

# A simply python controller for Baxter compatible with GPS (TODO)


import rospy

import baxter_interface
import baxter_external_devices

from aml_robot import baxter_robot

from aml_io.io_tools import save_data, load_data


from std_msgs.msg import (
    UInt16,
)

from baxter_interface import CHECK_VERSION

from aml_perception import camera_sensor
from functools import partial
import cv2

import numpy as np

def callback(agent,msg):
	agent.current_state = msg
	quat = agent.current_state['ee_ori']
	agent.current_state['ee_ori'] = [quat.x, quat.y, quat.z, quat.w]


class AgentProxy(object):
	def __init__(self):
		self.current_state = None


def main():

	rospy.init_node('baxter_test', anonymous=True)

	_rs = baxter_interface.RobotEnable(CHECK_VERSION)
	_rs.enable()


	agent = AgentProxy()

	right_limb = baxter_robot.BaxterArm('right',partial(callback,agent))
	right_limb.set_command_timeout(0.6)

	rate = rospy.Rate(150) # 10hz
	n_steps = 3000
	reset_after = 100
	save_every = 10
	save_count = 0
	data  = []


	state0 = None

	while state0 is None or state0['rgb_image'] is None:
		state0 = agent.current_state

	reset_pos = state0['position']

	loop_count = 0
	quit = False

	while not rospy.is_shutdown() and save_count < n_steps and not quit:
		if agent.current_state and loop_count%save_every == 0:

			data.append(agent.current_state)
			print("save_count: ",save_count)
			
			cv2.imshow("RGB Image window", agent.current_state['rgb_image'])
			save_count += 1



		if save_count%reset_after == 0:
			print("Reseting to initial joint positions")
			right_limb.move_to_joint_pos(reset_pos)

		cmd = np.random.randn(7)#*np.sqrt(0.95)
		right_limb.exec_velocity_cmd(cmd)

		loop_count += 1

		
		key = cv2.waitKey(1)
		quit = key == 27

		rate.sleep()


	save_data(data,'data_std1.pkl')



if __name__ == "__main__":
	main()


	

