import numpy as np
import quaternion

import copy

import rospy

from config import OS_VELCTY_CNTLR
from aml_ctrl.controllers.os_controller import OSController

from aml_ctrl.utilities.utilities import quatdiff

class OSVelocityController(OSController):
    def __init__(self, robot_interface, config = OS_VELCTY_CNTLR):

        OSController.__init__(self, robot_interface, config)

        #proportional gain for position
        self._kp_p       = self._config['kp_p']
        #derivative gain for position
        self._kd_p       = self._config['kd_p']
        #proportional gain for orientation
        self._kp_o       = self._config['kp_o']
        #derivative gain for orientation
        self._kd_o       = self._config['kd_o']
        #proportional gain for null space controller
        self._null_kp  = self._config['null_kp']
        #derivative gain for null space controller
        self._null_kd  = self._config['null_kd']
        #null space control gain
        self._alpha    = self._config['alpha']

        self._integrate_jnt_velocity = self._config['integrate_jnt_velocity']

        self._deactivate_wait_time = self._config['deactivate_wait_time']

        
        self._qr_old   = self._robot._state['position']

        self._dqr_old  = np.zeros(self._robot._nu)

        if 'rate' in self._config:
            self._rate = rospy.timer.Rate(self._config['rate'])

    def compute_cmd(self, time_elapsed):

        # calculate the Jacobian for the end effector

        goal_pos       = self._goal_pos

        goal_ori       = self._goal_ori

        goal_vel       = self._goal_vel

        goal_omg       = self._goal_omg

        robot_state    = self._state

        q              = robot_state['position']

        dq             = robot_state['velocity']

        h              = robot_state['gravity_comp']

        # calculate the jacobian of the end effector
        jac_ee         = robot_state['jacobian']

        # calculate the inertia matrix in joint space
        Mq             = robot_state['inertia']

        # calculate position of the end-effector
        ee_pos         = robot_state['ee_point']
        ee_ori         = robot_state['ee_ori']

        curr_vel       = robot_state['ee_vel']
        curr_omg       = robot_state['ee_omg']

        #pseudo inverse of jacobian
        jac_star       = np.dot(jac_ee.T, (np.linalg.inv(np.dot(jac_ee, jac_ee.T))))

        #gradient of redundancy resolution function
        grad_g         = 

        curr_time      = rospy.Time.now(0)

        dt             = curr_time - self._old_time

        self._old_time = curr_time


        #reference velocity
        dxr            = goal_vel + self._kp_p(goal_pos-ee_pos)

        #reference_joint_velocity
        if self._integrate_jnt_velocity:
            
            dqr        = np.dot(jac_star, dxr) + self._alpha*(np.eye(self._robot._nu) - np.dot(np.dot(jac_star, jac_ee), grad_g))

        else:

            dqr        = np.zeros_like(dq)


        ddqr           = (dqr - self._dqr_old)/dt

        self._dqr_old  = dqr

        qr             = self._qr_old + dqr*dt

        self._qr_old   = qr

        


        # convert the mass compensation into end effector space
        Mx_inv         = np.dot(jac_ee, np.dot(np.linalg.inv(Mq), jac_ee.T))
        svd_u, svd_s, svd_v = np.linalg.svd(Mx_inv)

        # cut off any singular values that could cause control problems
        singularity_thresh  = .00025
        for i in range(len(svd_s)):
            svd_s[i] = 0 if svd_s[i] < singularity_thresh else \
                1./float(svd_s[i])

        # numpy returns U,S,V.T, so have to transpose both here
        # convert the mass compensation into end effector space
        Mx   = np.dot(svd_v.T, np.dot(np.diag(svd_s), svd_u.T))

        # x_des   = goal_pos - ee_pos
        x_des   = self._kp_p*(goal_pos - ee_pos) + self._kd_p*(goal_vel - curr_vel)
 
        if self._orientation_ctrl:
            if goal_ori is None:
                print "For orientation control, pass goal orientation!"
                raise ValueError
            else:
                if type(goal_ori) is np.quaternion:
                    omg_des  = quatdiff(quaternion.as_float_array(goal_ori)[0], quaternion.as_float_array(ee_ori)[0])
                elif len(goal_ori) == 3:
                    # omg_des = goal_ori
                    omg_des = self._kp_o*goal_ori + self._kd_o*(goal_omg - curr_omg)
                else:
                    print "Wrong dimension"
                    raise ValueError
        else:
            omg_des = np.zeros(3)

        #print "h: ", h
        a_g                 = -np.dot(np.dot(jac_ee, np.linalg.inv(Mq)), h)
 
        # calculate desired force in (x,y,z) space
        Fx                  = np.dot(Mx, np.hstack([x_des, omg_des]) + 0.*a_g)


        # transform into joint space, add vel and gravity compensation
        u                   = np.dot(jac_ee.T, Fx)

        # calculate our secondary control signa
        # calculated desired joint angle acceleration

        prop_val            = (self._robot.q_mean - q)#((q_mean - q) + np.pi) % (np.pi*2) - np.pi

        q_des               = (self._null_kp * prop_val - self._null_kd * dq).reshape(-1,)

        u_null              = np.dot(Mq, q_des)

        # calculate the null space filter
        Jdyn_inv            = np.dot(Mx, np.dot(jac_ee, np.linalg.inv(Mq)))

        null_filter         = np.eye(len(q)) - np.dot(jac_ee.T, Jdyn_inv)

        u_null_filtered     = np.dot(null_filter, u_null)

        u                   += self._alpha*u_null_filtered

        if np.any(np.isnan(u)):
            u               = self._cmd
        else:
            self._cmd       = u

        # Never forget to update the error
        self._error = {'linear' : x_des, 'angular' : omg_des}

        return self._cmd

    def send_cmd(self,time_elapsed):
        self._robot.exec_torque_cmd(self._cmd)


    def set_active(self,is_active):

        OSController.set_active(self,is_active)

        if is_active is False:
            hold_time = rospy.Duration(self._deactivate_wait_time)
            last_time = rospy.Time.now()
            while (rospy.Time.now() - last_time) <= hold_time:
                self._robot.exec_position_cmd_delta(np.zeros(self._robot._nu))