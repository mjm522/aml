import abc
import copy
import rospy
from config import CONTROLLER

class ClassicalController(object):
	__metaclass__ = abc.ABCMeta

	def __init__(self, robot_interface):
		config = copy.deepcopy(CONTROLLER)
    	self._rate = rospy.timer.Rate(config['rate'])

	@abc.abstractmethod
	def compute_cmd(self, goal_pos, goal_ori, orientation_ctrl):
		raise NotImplementedError("Must be implemented in the subclass")

	def update_cmd(self):
		return self._cmd

	def send_cmd(self):
		raise NotImplementedError("Must be implemented in the subclass")

	def sleep(self):
		self._rate.sleep()


