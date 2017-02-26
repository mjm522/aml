import numpy as np
import quaternion
import copy
import rospy
from config import OS_JT_TORQUE_CNTLR
from aml_ctrl.utilities.utilities import quatdiff
from aml_ctrl.controllers.os_controller import OSController

class OSJTTorqueController(OSController):
    def __init__(self, robot_interface, config = OS_JT_TORQUE_CNTLR):

        OSController.__init__(self,robot_interface, config)

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

        self._pos_threshold = self._config['pos_threshold']


    def compute_cmd(self, time_elapsed):
    
        goal_pos       = self._goal_pos

        goal_ori       = self._goal_ori

        robot_state    = self._robot._state

        q              = robot_state['position']

        #for position commands sending zeros would not good. 
        self._cmd      = np.zeros_like(q)

        dq             = robot_state['velocity']

        # calculate the jacobian of the end effector
        jac_ee         = robot_state['jacobian']

        curr_pos, curr_ori  = self._robot.get_ee_pose()
        curr_vel, curr_omg  = self._robot.get_ee_velocity()

        delta_pos      = goal_pos - curr_pos

        # delta_vel      =  - curr_vel


        if self._orientation_ctrl:
            if goal_ori is None:
                print "For orientation control, pass goal orientation!"
                raise ValueError

            delta_ori       = quatdiff(quaternion.as_float_array(goal_ori)[0], quaternion.as_float_array(curr_ori)[0])

            delta           = np.hstack([self._kp_p*delta_pos, 
                                         self._kp_o*delta_ori])
        else:

            jac_ee          = jac_ee[0:3,:]
            delta_ori       = None
            delta           = self._kp_p*delta_pos 

        # jac_star            = np.dot(jac_ee.T, (np.linalg.inv(np.dot(jac_ee, jac_ee.T))))

        # prop_val            = (self._robot.q_mean - q) #+ np.pi) % (np.pi*2) - np.pi

        # q_null              = (self._null_kp * prop_val - self._null_kd * dq).reshape(-1,)

        # u_null              = self._alpha*np.dot((np.eye(self._robot._nu) - np.dot(jac_star,jac_ee)), q_null)

        # u_err               = np.dot(jac_star, delta)

        # self._cmd           = (u_null + u_err)*self._dt

        self._cmd           = np.dot(-delta, jac_ee)

        if np.any(np.isnan(self._cmd)) or np.linalg.norm(delta_pos) < self._pos_threshold:
            self._cmd       = np.zeros(self._robot._nu)
            
        # Never forget to update the error
        self._error = {'linear' : delta_pos, 'angular' : delta_ori}

        return self._cmd


    def send_cmd(self,time_elapsed):
        # self._robot.exec_position_cmd(self._cmd)
        # self._robot.exec_position_cmd2(self._cmd)
        self._robot.exec_torque_cmd(self._cmd)