import numpy as np
import quaternion

import copy

import rospy

from config import JS_TORQUE_CNTLR
from aml_ctrl.controllers.js_controller import JSController

from aml_ctrl.utilities.utilities import quatdiff

class JSTorqueController(JSController):
    def __init__(self, robot_interface, config = JS_TORQUE_CNTLR):

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

        self._dq = np.zeros_like(self._goal_js_pos)

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


        robot_state    = self._state

        q              = robot_state['position']

        dq             = self._dq*0.99 + robot_state['velocity']*0.01
        self._dq = dq

        if np.linalg.norm(dq) < 1e-3:
            dq = np.zeros_like(q)

        h              = robot_state['gravity_comp']

        # calculate the jacobian of the end effector
        jac_ee         = robot_state['jacobian']

        # calculate the inertia matrix in joint space
        Mq             = robot_state['inertia']

        js_delta       = goal_js_pos-q

        u              = np.dot(Mq, self._kp_q*js_delta + self._kd_dq*(goal_js_vel-dq))
 
        # calculate our secondary control signa
        # calculated desired joint angle acceleration

        prop_val            = (self._robot.q_mean - q)#((q_mean - q) + np.pi) % (np.pi*2) - np.pi

        q_des               = (self._null_kp * prop_val - self._null_kd * dq).reshape(-1,)

        u_null              = np.dot(Mq, q_des)

        # calculate the null space filter
        null_filter         = np.eye(len(q)) - np.dot(jac_ee.T, np.linalg.pinv(jac_ee.T))

        u_null_filtered     = np.dot(null_filter, u_null)

        u                   += self._alpha*u_null_filtered

        if np.any(np.isnan(u)):
            u               = self._cmd
        else:
            self._cmd       = u


        # Never forget to update the error
        self._error = {'js_pos': js_delta}

        return self._cmd

    def send_cmd(self,time_elapsed):
        self._robot.exec_torque_cmd(self._cmd)


    def set_active(self,is_active):

        JSController.set_active(self,is_active)

        # if is_active is False:
        #     hold_time = rospy.Duration(self._deactivate_wait_time)
        #     last_time = rospy.Time.now()
        #     while (rospy.Time.now() - last_time) <= hold_time:
        #         self._robot.exec_position_cmd_delta(np.zeros(self._robot._nu))
