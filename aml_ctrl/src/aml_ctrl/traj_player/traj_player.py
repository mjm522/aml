import rospy
import numpy as np



class TrajPlayer():

    def __init__(self, robot_interface, controller, trajectory, timeout=1.0, rate=100):

        self._ctrlr =  controller(robot_interface=robot_interface)

        self._ctrlr.set_active(True)

        self._traj = trajectory

        self._rate = rospy.Rate(rate)

        self._time_steps = len(trajectory[trajectory.keys()[0]])

        self._timeout = timeout

    def player(self):

        finished = False

        t = 0

        while not finished:

            if self._ctrlr.type is 'os':

                self._ctrlr.set_goal(  goal_pos=self._traj['pos_traj'][t], 
                                       goal_ori=self._traj['ori_traj'][t], 
                                       goal_vel=self._traj['vel_traj'][t], 
                                       goal_omg=self._traj['omg_traj'][t], 
                                       orientation_ctrl = True)

                lin_error, ang_error, success, time_elapsed = self._ctrlr.wait_until_goal_reached(timeout=self._timeout)

            elif self._ctrlr.type is 'js':

                self._ctrlr.set_goal(  goal_js_pos=self._traj['pos_traj'][t], 
                                       goal_js_vel=self._traj['vel_traj'][t], 
                                       goal_js_acc=self._traj['acc_traj'][t])
            
            
                js_pos_error, success, time_elapsed = self._ctrlr.wait_until_goal_reached(timeout=self._timeout)

            t += 1
            
            finished = (t >= self._time_steps)

            self._rate.sleep()

        if self._ctrlr.type is 'os':

            lin_error, ang_error, success, time_elapsed = self._ctrlr.wait_until_goal_reached(timeout=5)
        
        elif self._ctrlr.type is 'js':
            
            js_pos_error, success, time_elapsed = self._ctrlr.wait_until_goal_reached(timeout=5.0)

        self._ctrlr.set_active(False)



