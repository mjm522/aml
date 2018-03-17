import rospy
import numpy as np


class TrajPlayer():

    def __init__(self, robot_interface, controller, trajectory, timeout=1.0, total_timeout=None, rate=100):

        self._ctrlr =  controller(robot_interface=robot_interface)

        self._ctrlr.set_active(True)

        self._traj = trajectory

        self._rate = rospy.Rate(rate)

        self._time_steps = len(trajectory['pos_traj'])

        self._timeout = timeout

        self._total_timeout = total_timeout

    def player(self):

        finished = False

        start = rospy.get_time()

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

                if self._traj['vel_traj'] is None:
                    goal_js_vel = None
                else:
                    goal_js_vel = self._traj['vel_traj'][t]

                if self._traj['acc_traj'] is None:
                    goal_js_acc = None
                else:
                    goal_js_acc = self._traj['acc_traj'][t]

                self._ctrlr.set_goal(  goal_js_pos=self._traj['pos_traj'][t],
                                       goal_js_vel=goal_js_vel,
                                       goal_js_acc=goal_js_acc)


                js_pos_error, success, time_elapsed = self._ctrlr.wait_until_goal_reached(timeout=self._timeout)

            t += 1

            timed_out = self._total_timeout is not None and rospy.get_time()-start > self._total_timeout

            finished = bool(t >= self._time_steps or timed_out)

            self._rate.sleep()

        if self._ctrlr.type is 'os' and not timed_out:

            lin_error, ang_error, success, time_elapsed = self._ctrlr.wait_until_goal_reached(timeout=5)

        elif self._ctrlr.type is 'js' and not timed_out:

            js_pos_error, success, time_elapsed = self._ctrlr.wait_until_goal_reached(timeout=1.0)

        self._ctrlr.set_active(False)

        return timed_out



