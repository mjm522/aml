import numpy as np

def convert_rospy_time2sec(rospy_time):

	return rospy_time.secs + rospy_time.nsecs*1e-9