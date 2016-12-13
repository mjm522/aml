import numpy as np
import copy
import rospy
from config import OSC_TORQUE_CNTLR
from aml_ctrl.classical_controller import ClassicalController

class OSC_TorqueController(ClassicalController):
    def __init__(self, robot_interface):
        self._robot    = robot_interface
        self._cmd      = np.zeros(self._robot._nu)

        config         = copy.deepcopy(OSC_TORQUE_CNTLR)

        ClassicalController.__init__(self, robot_interface)

        #proportional gain
        self._kp       = config['kp']
        #derivative gain
        self._kd       = config['kd']
        #proportional gain for null space controller
        self._null_kp  = config['null_kp']
        #derivative gain for null space controller
        self._null_kd  = config['null_kd']
        #null space control gain
        self._alpha    = config['alpha']

        if 'rate' in config:
            self._rate = rospy.timer.Rate(config['rate'])

    def compute_cmd(self, goal_pos, goal_ori, orientation_ctrl=False):

        # calculate the Jacobian for the end effector

        robot_state    = self._robot._state

        q              = robot_state['position']

        dq             = robot_state['velocity']

        h              = robot_state['gravity_comp']

        # calculate the jacobian of the end effector
        jac_ee         = robot_state['jacobian']

        # calculate the inertia matrix in joint space
        Mq             = robot_state['inertia']

        # calculate position of the end-effector
        ee_xyz, ee_ori = self._robot.get_ee_pose()

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

        x_des   = goal_pos - ee_xyz
 
        if orientation_ctrl:
            if goal_ori is None:
                print "For orientation control, pass goal orientation!"
                raise ValueError
            else:
                if type(goal_ori) is np.quaternion:
                    omg_des  = quatdiff(quaternion.as_float_array(goal_ori)[0], quaternion.as_float_array(ee_ori)[0])
                elif len(goal_ori) == 3:
                    omg_des = goal_ori
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
        u                   = self._kp * np.dot(jac_ee.T, Fx) - np.dot(Mq, self._kd * dq)

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

        return self._cmd

    def send_cmd(self):
        self._robot.exec_torque_cmd(self._cmd)
        self._rate.sleep()