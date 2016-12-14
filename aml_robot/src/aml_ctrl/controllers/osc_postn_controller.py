import numpy as np
import quaternion
import copy
import rospy
from config import OSC_POSTN_CNTLR
from aml_ctrl.utilities.utilities import quatdiff
from aml_ctrl.classical_controller import ClassicalController

class OSCPositionController(ClassicalController):
    def __init__(self, robot_interface, config = OSC_POSTN_CNTLR):

        ClassicalController.__init__(self,robot_interface, config)

        print(config)
        #proportional gain
        self._kp       = self._config['kp']
        #derivative gain
        self._kd       = self._config['kd']
        #proportional gain for null space controller
        self._null_kp  = self._config['null_kp']
        #derivative gain for null space controller
        self._null_kd  = self._config['null_kd']
        #null space control gain
        self._alpha    = self._config['alpha']

        self._pos_threshold = self._config['pos_threshold']

        self._dt = self._config['dt']


    def compute_cmd(self,time_elapsed):

        
        goal_pos       = self._goal_pos

        goal_ori       = self._goal_ori
        
        robot_state    = self._robot._state

        q              = robot_state['position']

        #for position commands sending zeros would not good. 
        self._cmd      = q

        dq             = robot_state['velocity']

        # calculate the jacobian of the end effector
        jac_ee         = robot_state['jacobian']

        error                   = 100.

        curr_pos, curr_ori  = self._robot.get_ee_pose()


        delta_pos      = (goal_pos-curr_pos)


        if self._orientation_ctrl:
            if goal_ori is None:
                print "For orientation control, pass goal orientation!"
                raise ValueError

            delta_ori       = quatdiff(quaternion.as_float_array(goal_ori)[0], quaternion.as_float_array(curr_ori)[0])
            delta           = np.hstack([delta_pos, delta_ori])
        else:

            jac_ee          = jac_ee[0:3,:]
            delta           = delta_pos

        jac_star            = np.dot(jac_ee.T, (np.linalg.inv(np.dot(jac_ee, jac_ee.T))))
        null_q              = self._kp*np.dot(jac_star, delta) + self._alpha*np.dot((np.eye(len(q)) - np.dot(jac_star,jac_ee)),(self._robot.q_mean - q))
        self._cmd           = q + null_q*self._dt

        if np.any(np.isnan(self._cmd)) or np.linalg.norm(delta_pos) < self._pos_threshold:
            self._cmd       = q


        # Never forget to update the error
        self._error = {'linear' : delta_pos, 'angular' : delta_ori}

        return self._cmd


    def send_cmd(self,time_elapsed):
        self._robot.exec_position_cmd(self._cmd)