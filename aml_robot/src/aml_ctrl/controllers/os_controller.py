import copy
import rospy
from aml_ctrl.controllers.config import OS_CONTROLLER
from aml_ctrl.controller import Controller

import numpy as np

class OSController(Controller):

    def __init__(self, robot_interface, config = OS_CONTROLLER):

        self._config  = copy.deepcopy(config)

        Controller.__init__(self, robot_interface, config)

        self._error = {'linear' : np.zeros(3), 'angular' : np.zeros(3)}

        self._orientation_ctrl = config['use_orientation_ctrl']
        self._lin_thr = config['linear_error_thr']
        self._ang_thr = config['angular_error_thr']

        self._goal_pos, self._goal_ori  =  self._robot.get_ee_pose()

        self._goal_vel  = np.zeros(3)
        self._goal_omg  = np.zeros(3)

        self._type = 'os'

    @property
    def type(self):
        return self._type

    def set_goal(self, goal_pos, goal_ori, goal_vel=None, goal_omg=None, orientation_ctrl = True):
        
        self._goal_pos = goal_pos
        self._goal_ori = goal_ori

        if goal_vel is not None:
            self._goal_vel = goal_vel

        if goal_omg is not None:
            self._goal_omg = goal_omg

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

            if self._orientation_ctrl:
                ang_error = np.linalg.norm(self._error['angular'])
            else:
                ang_error = None

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



