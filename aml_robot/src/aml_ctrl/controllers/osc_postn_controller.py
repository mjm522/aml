import numpy as np
import quaternion
import copy
import rospy
from config import OSC_POSTN_CNTLR
from aml_ctrl.utilities.utilities import quatdiff
from aml_ctrl.classical_controller import ClassicalController

class OSC_PostnController(ClassicalController):
    def __init__(self, robot_interface):
        self._robot    = robot_interface
        self._cmd      = np.zeros(self._robot._nu)

        config         = copy.deepcopy(OSC_POSTN_CNTLR)

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

        self._pos_threshold = config['pos_threshold']

        if 'rate' in config:
            self._rate = rospy.timer.Rate(config['rate'])

    def compute_cmd(self, goal_pos, goal_ori, orientation_ctrl=False):
        
        robot_state    = self._robot._state

        q              = robot_state['position']

        #for position commands sending zeros would not good. 
        self._cmd      = q

        dq             = robot_state['velocity']

        # calculate the jacobian of the end effector
        jac_ee         = robot_state['jacobian']

        error                   = 100.

        curr_pos, curr_ori  = self._robot.get_ee_pose()


        delta_pos      = goal_pos-curr_pos


        if orientation_ctrl:
            if goal_ori is None:
                print "For orientation control, pass goal orientation!"
                raise ValueError

            delta_ori       = quatdiff(quaternion.as_float_array(goal_ori)[0], quaternion.as_float_array(curr_ori)[0])
            delta           = np.hstack([delta_pos, delta_ori])
        else:

            jac_ee          = jac_ee[0:3,:]
            delta           = delta_pos

        jac_star            = np.dot(jac_ee.T, (np.linalg.inv(np.dot(jac_ee, jac_ee.T))))
        null_q              = self._alpha*np.dot(jac_star, delta) + np.dot((np.eye(len(q)) - np.dot(jac_star,jac_ee)),(self._robot.q_mean - q))
        self._cmd           = q + null_q

        if np.any(np.isnan(self._cmd)) or np.linalg.norm(delta_pos) < self._pos_threshold:
            self._cmd       = q

        return self._cmd


    def send_cmd(self):
        self._robot.exec_position_cmd(self._cmd)
        self._rate.sleep()