import rospy
import numpy as np

class TrajPlayer():

    def __init__(self, robot_interface, controller, trajectory):

        self._ctrlr =  controller(robot_interface=robot_interface)

        self._ctrlr.set_active(True)

        self._traj = trajectory

        self._rate = rospy.Rate(10)

        self._time_steps = len(trajectory[trajectory.keys()[0]])

    def player(self):

        finished = False

        t = 0

        while not finished:

            self._ctrlr.set_goal(  goal_pos=self._traj['pos_traj'][t], 
                                   goal_ori=self._traj['ori_traj'][t], 
                                   goal_vel=self._traj['vel_traj'][t], 
                                   goal_omg=self._traj['omg_traj'][t], 
                                   orientation_ctrl = True)

            lin_error, ang_error, success, time_elapsed = self._ctrlr.wait_until_goal_reached(timeout=1.0)

            t += 1
            
            finished = (t == self._time_steps)

            self._rate.sleep()

        lin_error, ang_error, success, time_elapsed = self._ctrlr.wait_until_goal_reached(timeout=5)
        self._ctrlr.set_active(False)



