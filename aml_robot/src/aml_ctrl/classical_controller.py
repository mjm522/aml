import abc
import copy
import rospy
from aml_ctrl.controllers.config import CONTROLLER

import numpy as np

class ClassicalController(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, robot_interface, config = CONTROLLER):

        self._config = copy.deepcopy(config)

        self._rate = rospy.timer.Rate(config['rate'])

        self._robot    = robot_interface
        self._cmd      = np.zeros(self._robot._nu)

        update_period = rospy.Duration(1.0/config['rate'])
        rospy.Timer(update_period, self.update)

        self._last_time = rospy.Time.now()

        self._error = {'linear' : np.zeros(3), 'angular' : np.zeros(3), 'js_pos' : np.zeros(self._robot._nu)}

        #for operation space controller
        self._orientation_ctrl = config['use_orientation_ctrl']
        self._lin_thr = config['linear_error_thr']
        self._ang_thr = config['angular_error_thr']

        #for joint space controller
        self._js_thr = config['js_pos_error_thr']

        self._goal_pos, self._goal_ori  =  self._robot.get_ee_pose()

        self._goal_vel  = np.zeros(3)
        self._goal_omg  = np.zeros(3)

        self._goal_js_pos = self._robot._state['position']
        self._goal_js_vel = self._robot._state['velocity']
        self._goal_js_acc = np.zeros_like(self._goal_js_pos)

        self._is_active = False

    @abc.abstractmethod
    def compute_cmd(self,time_elapsed):
        raise NotImplementedError("Must be implemented in the subclass")

    def update(self,event):

        # If this controller is not active, then we do nothing and return immediately
        if not self._is_active:
            return

        time_elapsed = event.current_real - self._last_time
        
        self._last_time = event.current_real


        # Read state

        self._state = self._robot._state

        # Update command

        self._cmd = self.compute_cmd(time_elapsed)

        # Send command to be executed by the robot

        self.send_cmd(time_elapsed)

    #this is for operation space controller
    def set_goal(self, goal_pos, goal_ori, goal_vel=None, goal_omg=None, orientation_ctrl = True):
        self._goal_pos = goal_pos
        self._goal_ori = goal_ori

        if goal_vel is not None:
            self._goal_vel = goal_vel

        if goal_omg is not None:
            self._goal_omg = goal_omg

        self._orientation_ctrl = orientation_ctrl
        self._has_reached_goal = False

    #this is for joint space controller
    # def set_goal(self, goal_js_pos, goal_js_vel, goal_js_acc):
        
    #     self._goal_js_pos = goal_js_pos
    #     self._goal_js_vel = goal_js_vel
    #     self._goal_js_acc = goal_js_acc

    #     self._has_reached_goal = False


    # Wait timeout seconds for reaching the last set goal
    def wait_until_goal_reached(self, timeout = 5.0, jsc=False):

        timeout = rospy.Duration(timeout)
        reached_goal = False
        failed = False
        rate = rospy.Rate(100)
        time_start = rospy.Time.now()
        lin_error = 0.0
        ang_error = 0.0
        while not reached_goal and not failed:

            if jsc:
                js_error = np.linalg.norm(self._error['js_pos'])

            else:
                lin_error = np.linalg.norm(self._error['linear'])

            if self._orientation_ctrl:
                ang_error = np.linalg.norm(self._error['angular'])
            else:
                ang_error = None

            if jsc:
                if js_error <= self._js_thr:
                    reached_goal = True
            else:
                if lin_error <= self._lin_thr and ang_error <= self._ang_thr:
                    reached_goal = True

            time_elapsed = rospy.Time.now() - time_start

            if time_elapsed >= timeout and not reached_goal:
                failed = True

            # print("lin_error: %0.4f ang_error: %0.4f elapsed_time: "%(lin_error,ang_error),time_elapsed.secs)
            rate.sleep()



        success = reached_goal and not failed

        # print("lin_error: %0.4f ang_error: %0.4f elapsed_time: "%(lin_error,ang_error),time_elapsed, " success: ", success)

        return lin_error, ang_error, success, time_elapsed


    def send_cmd(self,time_elapsed):
        raise NotImplementedError("Must be implemented in the subclass")

    def sleep(self):
        self._rate.sleep()


    def set_active(self,is_active):
        self._is_active = is_active



