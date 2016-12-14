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

        self._error = {'linear' : np.zeros(3), 'angular' : np.zeros(3)}

        self._orientation_ctrl = config['use_orientation_ctrl']

        self._lin_thr = config['linear_error_thr']
        self._ang_thr = config['angular_error_thr']

        self._goal_pos, self._goal_ori  =  self._robot.get_ee_pose()

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

    def set_goal(self,goal_pos,goal_ori,orientation_ctrl = True):
        self._goal_pos = goal_pos
        self._goal_ori = goal_ori
        self._orientation_ctrl = orientation_ctrl
        self._has_reached_goal = False


    # Wait timeout seconds for reaching the last set goal
    def wait_until_goal_reached(self, timeout = 5.0):

        timeout = rospy.Duration(timeout)
        reached_goal = False
        failed = False
        rate = rospy.Rate(100)
        time_start = rospy.Time.now()
        lin_error = 0.0
        ang_error = 0.0
        while not reached_goal and not failed:
            lin_error = np.linalg.norm(self._error['linear'])
            ang_error = np.linalg.norm(self._error['angular'])



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



