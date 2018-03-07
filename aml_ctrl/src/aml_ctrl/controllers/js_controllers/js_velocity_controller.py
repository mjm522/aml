import numpy as np
import quaternion

import copy

import rospy

from config import JS_VELOCITY_CNTLR_BAXTER as JS_VELOCITY_CNTLR
from aml_ctrl.controllers.js_controller import JSController

from aml_ctrl.utilities.utilities import quatdiff

class JSVelocityController(JSController):
    def __init__(self, robot_interface, config = JS_VELOCITY_CNTLR):

        JSController.__init__(self, robot_interface, config)

        #proportional gain for position
        self._kp_q        = self._config['kp_q']
        #derivative gain for position
        self._kd_dq       = self._config['kd_dq']

        #proportional gain for null space controller
        self._null_kp  = self._config['null_kp']
        #derivative gain for null space controller
        self._null_kd  = self._config['null_kd']
        #null space control gain
        self._alpha    = self._config['alpha']

        self._deactivate_wait_time = self._config['deactivate_wait_time']

        self._alpha = self._config['velocity_filter_alpha']

        self._dq = np.zeros_like(self._goal_js_pos)

        if 'rate' in self._config:
            self._rate = rospy.timer.Rate(self._config['rate'])

    def compute_cmd(self, time_elapsed):

        goal_js_pos       = self._goal_js_pos

        if self._goal_js_vel is None:

            goal_js_vel = np.zeros_like(goal_js_pos)

        else:

            goal_js_vel   = self._goal_js_vel

        if self._goal_js_acc is None:

            goal_js_acc = np.zeros_like(goal_js_pos)

        else:

            goal_js_acc       = self._goal_js_acc

        robot_state    = self._state

        q              = robot_state['position']

        dq             = self._dq*(1.0 - self._alpha) + robot_state['velocity']*self._alpha

        self._dq = dq

        js_delta       = goal_js_pos-q

        if np.linalg.norm(dq) < 1e-3:
            dq = np.zeros_like(q)

        if np.linalg.norm(js_delta) < 1e-3:
            js_delta = np.zeros_like(q)

        u              = self._kp_q*js_delta + self._kd_dq*(goal_js_vel - dq)

        if np.any(np.isnan(u)):
            u               = self._cmd
        else:
            self._cmd       = u

        # Never forget to update the error
        self._error = {'js_pos': js_delta}

        return self._cmd

    def send_cmd(self,time_elapsed):
        self._robot.exec_velocity_cmd(self._cmd) 


    def set_active(self,is_active):

        JSController.set_active(self,is_active)
