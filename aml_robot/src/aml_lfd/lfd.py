import abc
import copy
import rospy
import numpy as np

class LfD(object):
	__metaclass__ = abc.ABCMeta

	def __init__(self, config):
		pass

	@abc.abstractmethod
	def encode_demo(self):
		raise NotImplementedError("Must be implemented in the subclass")

