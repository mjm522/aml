import abc
import copy
import rospy
from aml_ctrl.config import CONTROLLER

import numpy as np

class Controller(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, robot_interface, config = CONTROLLER):

        self._config = copy.deepcopy(config)

        self._rate = rospy.timer.Rate(config['rate'])

        self._robot    = robot_interface
        self._cmd      = np.zeros(self._robot._nu)

        update_period = rospy.Duration(1.0/config['rate'])
        rospy.Timer(update_period, self.update)

        self._last_time = rospy.Time.now()

        self._error = None

        self._is_active = False

    @abc.abstractmethod
    def compute_cmd(self,time_elapsed):
        raise NotImplementedError("Must be implemented in the subclass")

    def update(self,event):

        # If this controller is not active, then we do nothing and return immediately
        if not self._is_active:
            return

        time_elapsed = event.current_real - self._last_time
        
        self._last_time = event.current_real


        # Read state

        self._state = self._robot._state

        # Update command

        self._cmd = self.compute_cmd(time_elapsed)

        # Send command to be executed by the robot

        self.send_cmd(time_elapsed)

    @abc.abstractmethod
    def wait_until_goal_reached(self, timeout):
        raise NotImplementedError("Must be implemented in the subclass")

    @abc.abstractmethod
    def send_cmd(self,time_elapsed):
        raise NotImplementedError("Must be implemented in the subclass")

    def sleep(self):
        self._rate.sleep()

    def set_active(self,is_active):
        self._is_active = is_active



