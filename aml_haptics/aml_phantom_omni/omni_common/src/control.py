#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid
from omni_msgs.msg import OmniFeedback
from geometry_msgs.msg import PoseStamped
import numpy as np
from pid_controller.pid import PID
import tf2_ros

mapMsg = None
poseMsg = None
pid = PID(p=100, i=50, d=0.0)
pid._target = -0.01

def MapCallback(msg):
	global mapMsg
	mapMsg = np.asarray(msg.data)
	mapMsg = mapMsg.reshape(msg.info.height, msg.info.width)

def PoseCallback(msg):
	global poseMsg
	poseMsg = msg.pose

def run():
	rospy.init_node('phantom_controll', anonymous=False)
	rospy.Subscriber('/map', OccupancyGrid, MapCallback)
	rospy.Subscriber('/phantom/pose', PoseStamped, PoseCallback)
	force_pub = rospy.Publisher('/phantom/force_feedback', OmniFeedback, queue_size=10)

	rate = rospy.Rate(100) # 5 hz

	print "[Map Example Started]"

	global mapMsg, poseMsg
	forceMsg = OmniFeedback()

	while not rospy.is_shutdown():
		if mapMsg == None or poseMsg == None:
			rate.sleep()
			print "Waiting for map"
			continue

		map_x = int((poseMsg.position.x)/0.002 + 50)
		map_y = int((poseMsg.position.y)/0.002 + 50)
		forceMsg.force.x = 0.0
		forceMsg.force.y = 0.0
		forceMsg.force.z = 0.0

		out = False
		if map_x < 0:
			out = True
			forceMsg.force.x = -1
		elif map_x >= 100: 
			out = True
			forceMsg.force.x = 1
		if map_y < 0:
			out = True
			forceMsg.force.z = 1
		elif map_y >= 100: 
			out = True
			forceMsg.force.z = -1
		if out:
			force_pub.publish(forceMsg)
			continue

		if mapMsg[map_y][map_x] == 100:
			pid._target = -0.017
		else:
			pid._target = -0.007

		forceMsg.force.y = pid(feedback=-poseMsg.position.z)
		forceMsg.force.y = min(max(forceMsg.force.y, -5.0), 10.0)

		force_pub.publish(forceMsg)
		rate.sleep()

if __name__ == '__main__':
	try:
		run()
	except rospy.ROSInterruptException:
		pass