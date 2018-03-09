import abc
import copy
import rospy
import numpy as np

from aml_teleop.config import TELEOP

class TeleOp(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, master_interface, slave_interface, config = TELEOP):

        pass

    @abc.abstractmethod
    def compute_cmd(self, time_elapsed):
        
        raise NotImplementedError("Must be implemented in the subclass")


            