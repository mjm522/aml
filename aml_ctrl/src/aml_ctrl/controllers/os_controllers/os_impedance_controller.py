import copy
import rospy
import quaternion
import numpy as np
from config import OS_IMPEDANCE_CNTLR
from aml_ctrl.controllers.os_controller import OSController
from aml_ctrl.utilities.utilities import quatdiff, pseudo_inv


class OSImpedanceController(OSController):

    def __init__(self, robot_interface, config = OS_IMPEDANCE_CNTLR):

        OSController.__init__(self,robot_interface, config)
        #proportional gain for position
        self._kp_p       = self._config['kp_p']
        #derivative gain for position
        self._kd_p       = self._config['kd_p']
        #desired damping in the joint space
        self._kd_q       = self._config['kd_q']
        #desired task space impedance
        self._Md         =  self._config['Md']
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

        self._use_ori_control = self._config['use_orientation_ctrl']

        self.initialise()


    def get_time(self):

        time_now       = rospy.Time.now()
        return time_now.secs + time_now.nsecs*1e-9


    def initialise(self):

        robot_state    = self._robot._state
        self._t_old    = self.get_time()
        
        if self._use_ori_control:
            self._Jee_old  = robot_state['jacobian']
        else:
            self._Jee_old  = robot_state['jacobian'][:3,:]


    def compute_cmd_1(self, time_elapsed):

        t              = self.get_time()

        dt             = t - self._t_old

        if dt == 0.0:
            dt = 0.001

        self._t_old    = t

        robot_state    = self._robot._state

        q              = robot_state['position']

        dq             = robot_state['velocity']

        # calculate the jacobian of the end effector
        Jee            = robot_state['jacobian'][:3,:]

        Mq             = robot_state['inertia']

        if 'ee_force' in robot_state.keys():
            ee_force   = robot_state['ee_force']
        else:
            ee_force   = np.zeros(3)

        if 'ee_torque' in robot_state.keys():
            ee_torque  = robot_state['ee_torque']
        else:
            ee_torque  = np.zeros(3)

        curr_pos, curr_ori  = self._robot.get_ee_pose()
        curr_vel, curr_omg  = self._robot.get_ee_velocity()

        delta_pos      = self._goal_pos - curr_pos
        delta_vel      = self._goal_vel - curr_vel
        delta_ori      = quatdiff(self._goal_ori, curr_ori)
        delta_omg      = np.zeros(3)

        self._cmd      = np.zeros_like(q)

        #Compute the derivative of jacobian matrix
        Jee_delta      = Jee - self._Jee_old
        Jee_dot        = Jee_delta / dt
        self._Jee_old  = Jee

        #Compute cartesian space inertia matrix
        Mq_inv    = np.linalg.inv(Mq)
        Mcart_inv = np.dot(np.dot(Jee, Mq_inv), Jee.transpose())
        Mcart     = np.linalg.pinv(Mcart_inv, rcond=1e-3)

        #inertia shaping, as same as eee inertia
        Md_inv  = Mcart_inv #(np.linalg.inv(self._Md))

        #Compute dynamiclly consistent generlized inverse matrix and projection matrix
        Jbar_transpose = np.dot(np.dot(Mcart, Jee), Mq_inv)
        null_proj = np.eye(len(q)) - np.dot(Jee.transpose(), Jbar_transpose)

        #secondary pose torque
        tau_pose = np.zeros_like(q) - np.dot(self._kd_q, dq)
        tau_pose = np.dot(null_proj, tau_pose)

        tau_residual = np.dot(np.linalg.pinv(Jbar_transpose, rcond=1e-3), ee_force)

        #compute task torque
        #this from the paper: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6942847
        tau_task = -np.dot( np.dot(Jee.transpose(), Mcart), 
                    (np.dot(Jee_dot, dq) +  np.dot(Md_inv, (np.dot(self._kd_p, curr_vel) - ee_force )) ) ) - tau_residual
                
        
        #from morteza slide
        xdd = np.zeros(3)
        tmp = np.dot(Mcart, Md_inv)
        f = ee_force + np.dot(Mcart, xdd) + np.dot(tmp, (np.dot(self._kp_p, delta_pos) + np.dot(self._kd_p, delta_vel)) ) - np.dot(tmp, ee_force)
        tau_task = np.dot( np.dot(Jee.transpose(), Mcart),  f)

        self._cmd = tau_task + tau_pose

        return self._cmd

    def compute_cmd_2(self, time_elapsed):

        t              = self.get_time()

        dt             = t - self._t_old

        self._t_old    = t

        robot_state    = self._robot._state

        q              = robot_state['position']

        dq             = robot_state['velocity']

        # calculate the jacobian of the end effector
        Jee            = robot_state['jacobian'][:3,:]

        Mq             = robot_state['inertia']

        if 'ee_force' in robot_state.keys():
            ee_force   = robot_state['ee_force']
        else:
            ee_force   = np.zeros(3)

        if 'ee_torque' in robot_state.keys():
            ee_torque  = robot_state['ee_torque']
        else:
            ee_torque  = np.zeros(3)

        curr_pos, curr_ori  = self._robot.get_ee_pose()
        curr_vel, curr_omg  = self._robot.get_ee_velocity()

        delta_pos      = self._goal_pos - curr_pos
        delta_vel      = self._goal_vel - curr_vel
        delta_ori      = quatdiff(self._goal_ori, curr_ori)
        delta_omg      = np.zeros(3)

        self._cmd      = np.zeros_like(q)

        #Compute the derivative of jacobian matrix
        Jee_delta      = Jee - self._Jee_old
        Jee_dot        = Jee_delta / dt
        self._Jee_old  = Jee

        #Compute cartesian space inertia matrix
        Mq_inv    = np.linalg.inv(Mq)
        Mcart_inv = np.dot(np.dot(Jee, Mq_inv), Jee.transpose())
        Mcart     = np.linalg.pinv(Mcart_inv, rcond=1e-3)

        #inertia shaping, as same as eee inertia
        Md_inv  = Mcart_inv #(np.linalg.inv(self._Md))

        #Compute dynamiclly consistent generlized inverse matrix and projection matrix
        Jbar_transpose = np.dot(np.dot(Mcart, Jee), Mq_inv)
        null_proj = np.eye(len(q)) - np.dot(Jee.transpose(), Jbar_transpose)

        #secondary pose torque
        tau_pose = np.zeros_like(q) - np.dot(self._kd_q, dq)
        tau_pose = np.dot(null_proj, tau_pose)

        tau_residual = np.dot(np.linalg.pinv(Jbar_transpose, rcond=1e-3), ee_force)

        #compute task torque
        #this from the paper: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6942847
        tau_task = -np.dot( np.dot(Jee.transpose(), Mcart), 
                    (np.dot(Jee_dot, dq) +  np.dot(Md_inv, (np.dot(self._kd_p, curr_vel) - ee_force )) ) ) - tau_residual
                
        
        #from morteza slide
        xdd = np.zeros(3)
        tmp = np.dot(Mcart, Md_inv)
        f = ee_force + np.dot(Mcart, xdd) + np.dot(tmp, (np.dot(self._kp_p, delta_pos) + np.dot(self._kd_p, delta_vel)) ) - np.dot(tmp, ee_force)
        tau_task = np.dot( np.dot(Jee.transpose(), Mcart),  f)

        self._cmd = tau_task + tau_pose
        
        return self._cmd

    def compute_cmd(self, time_elapsed):
        return self.compute_cmd_1(time_elapsed)


    def send_cmd(self,time_elapsed):
        # self._robot.exec_position_cmd(self._cmd)
        # self._robot.exec_position_cmd2(self._cmd)
        self._robot.exec_torque_cmd(self._cmd)

