import numpy as np
import quaternion

import copy

import rospy

from config import JS_POSTN_CNTLR
from aml_ctrl.controllers.js_controller import JSController

from aml_ctrl.utilities.utilities import quatdiff

class JSPositionController(JSController):
    def __init__(self, robot_interface, config = JS_POSTN_CNTLR):

        JSController.__init__(self, robot_interface, config)

        self._deactivate_wait_time = self._config['deactivate_wait_time']

        if 'rate' in self._config:
            self._rate = rospy.timer.Rate(self._config['rate'])

    def compute_cmd(self, time_elapsed):

        # calculate the Jacobian for the end effector

        goal_js_pos       = self._goal_js_pos

        if self._goal_js_vel is None:

            goal_js_vel = np.zeros_like(goal_js_pos)

        else:

            goal_js_vel   = self._goal_js_vel

        if self._goal_js_acc is None:

            goal_js_acc = np.zeros_like(goal_js_pos)

        else:

            goal_js_acc       = self._goal_js_acc

        robot_state    = self._robot.state()

        q              = robot_state['position']

        dq             = robot_state['velocity']

        js_delta       = goal_js_pos-q

        u              = goal_js_pos

        if np.any(np.isnan(u)):
            u               = self._cmd
        else:
            self._cmd       = u

        # Never forget to update the error
        self._error = {'js_pos': js_delta}

        return self._cmd

    def send_cmd(self,time_elapsed):
        self._robot.exec_position_cmd(self._cmd) # changed to position cmd


    def set_active(self,is_active):

        JSController.set_active(self,is_active)
