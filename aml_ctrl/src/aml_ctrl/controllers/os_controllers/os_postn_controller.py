import numpy as np
import quaternion
import copy
import rospy
from config import OS_POSTN_CNTLR
from aml_ctrl.utilities.utilities import quatdiff
from aml_ctrl.controllers.os_controller import OSController

class OSPositionController(OSController):
    """
    This class is an implementation fo the Operational Space position control
    The type of control scheme is given in Robotics: Modelling, Planning and Control (page: 345)
    Some equation of this is also adpated from http://journals.sagepub.com/doi/abs/10.1177/0278364908091463
    Paper title : Operational Space Control: A Theoretical and Empirical Comparison
    This code control scheme it is assumed to have an already gravity compensated arm
    i.e. the gravity compensation happens in the joint space, while the operation space is used for task control
    """
    def __init__(self, robot_interface, config = OS_POSTN_CNTLR):
        """
        Constructor of the class,
        Args:
        robot_interface : interface to the arm (type: aml_robot)
        config: params: 
                        kp_p : proportional gain for position
                        kd_p : derivative gain for position
                        kp_o : proportional gain for orientation
                        kd_o : derivative gain for orientation
                        null_kp: proportional gain for null space controller
                        null_kd: derivative gain for null space controller
                        alpha: null space control mixing factor
                        rate: rate of speed sending commands
                        dt: time step
        """
        OSController.__init__(self,robot_interface, config)

        #proportional gain for position
        self._kp_p       = self._config['kp_p']
        #derivative gain for velocity
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

        self._pos_threshold = self._config['linear_error_thr']

        self._angular_threshold = self._config['angular_error_thr']

        self._dt = self._config['dt']


    def compute_cmd(self, time_elapsed):
    
        goal_pos       = self._goal_pos

        goal_ori       = self._goal_ori

        goal_vel       = self._goal_vel

        goal_omg       = self._goal_omg
        
        robot_state    = self._robot.state()

        q              = robot_state['position']

        #for position commands sending zeros would not good. 
        self._cmd      = q

        dq             = robot_state['velocity']

        # calculate the jacobian of the end effector
        jac_ee         = robot_state['jacobian']

        error          = 100.

        curr_pos, curr_ori  = self._robot.ee_pose()
        curr_vel, curr_omg  = self._robot.ee_velocity()

        delta_pos      = goal_pos - curr_pos

        delta_vel      = goal_vel - curr_vel


        if np.linalg.norm(delta_pos) < self._pos_threshold:
            delta_pos = np.zeros(delta_pos.shape)
            delta_vel = np.zeros(delta_vel.shape)

        if self._orientation_ctrl:
            if goal_ori is None:
                print "For orientation control, pass goal orientation!"
                raise ValueError

            delta_ori       = quatdiff(goal_ori, curr_ori)
            delta_omg       = goal_omg - curr_omg


            if np.linalg.norm(delta_ori) < self._angular_threshold:
                delta_ori = np.zeros(delta_ori.shape)
                delta_omg = np.zeros(delta_omg.shape)

            delta           = np.hstack([self._kp_p*delta_pos + self._kd_p*delta_vel, 
                                         self._kp_o*delta_ori + self._kd_o*delta_omg])
        else:

            jac_ee          = jac_ee[0:3,:]
            delta_ori       = None
            #compute PD control law
            delta           = self._kp_p*delta_pos + self._kd_p*delta_vel

        #compute pseudo inverse of paper: jacobian equation 5
        jac_star            = np.dot(jac_ee.T, (np.linalg.inv(np.dot(jac_ee, jac_ee.T))))

        #gradient of the secondary goal, paper : equation 9
        prop_val            = (self._robot.q_mean() - q) #+ np.pi) % (np.pi*2) - np.pi

        q_null              = (self._null_kp * prop_val - self._null_kd * dq).reshape(-1,)

        u_null              = self._alpha*np.dot((np.eye(self._robot._nu) - np.dot(jac_star,jac_ee)), q_null)

        u_err               = np.dot(jac_star, delta)

        self._cmd           = (u_null + u_err)*self._dt

        if np.any(np.isnan(self._cmd)): #or 
            self._cmd       = np.zeros(self._robot._nu)
            
        # Never forget to update the error
        self._error = {'linear' : delta_pos, 'angular' : delta_ori}

        return self._cmd


    def send_cmd(self,time_elapsed):
        self._robot.exec_position_cmd_delta(self._cmd)