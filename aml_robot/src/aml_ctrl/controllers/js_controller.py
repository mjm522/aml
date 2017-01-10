import copy
import rospy
from aml_ctrl.controllers.config import JS_CONTROLLER
from aml_ctrl.controller import Controller

import numpy as np

class JSController(Controller):

    def __init__(self, robot_interface, config = JS_CONTROLLER):

        self._config   = copy.deepcopy(config)

        Controller.__init__(self, robot_interface, config)

        self._error  = {'js_pos' : np.zeros(self._robot._nu)}
 
        self._js_thr = config['js_pos_error_thr']

        self._goal_js_pos = self._robot._state['position']
        self._goal_js_vel = self._robot._state['velocity']
        self._goal_js_acc = np.zeros_like(self._goal_js_pos)

        self._type = 'js'

    @property
    def type(self):
        return self._type

    def set_goal(self, goal_js_pos, goal_js_vel=None, goal_js_acc=None):
        
        self._goal_js_pos = goal_js_pos
        self._goal_js_vel = goal_js_vel
        self._goal_js_acc = goal_js_acc

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

            js_error = np.linalg.norm(self._error['js_pos'])

            if js_error <= self._js_thr:
                reached_goal = True

            time_elapsed = rospy.Time.now() - time_start

            if time_elapsed >= timeout and not reached_goal:
                failed = True

            # print("lin_error: %0.4f ang_error: %0.4f elapsed_time: "%(lin_error,ang_error),time_elapsed.secs)
            rate.sleep()

        success = reached_goal and not failed

        # print("lin_error: %0.4f ang_error: %0.4f elapsed_time: "%(lin_error,ang_error),time_elapsed, " success: ", success)

        return js_error, success, time_elapsed



