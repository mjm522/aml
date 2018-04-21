import rospy
import numpy as np

class RobotBulletInterface():

    def __init__(self, robot_interface, bullet_interface, rate=900):

        self._real_robot = robot_interface

        self._bullet_robot = bullet_interface

        self._master_jlims = self._real_robot._jnt_limits

        self._slave_jlims  = self._master_jlims

        self._num_master_jnts = len(self._master_jlims)

        self._num_slave_jnts = len(self._slave_jlims)

        self._rate = rate

        self._scale_from_home = False


    def master_js_scale(self):
        """
        this function finds the js scale of the master robot
        in a range 0 to 1
        """

        scale = np.zeros(self._num_master_jnts)

        curr_js = self._real_robot.state()['position']

        if self._scale_from_home:

            for k in range(self._num_master_jnts):

                scale[k] = (curr_js[k] - self._jnt_home[k])\
                /(self._master_jlims[k]['upper'] - self._master_jlims[k]['lower'])

            scale[scale > 0.5]  = 0.5

            scale[scale < -0.5] = -0.5

        else:

            for k in range(self._num_master_jnts):

                scale[k] = (curr_js[k] - self._master_jlims[k]['lower'])\
                         /(self._master_jlims[k]['upper'] - self._master_jlims[k]['lower'])

            scale[scale > 1.] = 1.

            scale[scale < 0.] = 0.

        return scale

    def compute_cmd(self):
        """
        once the scale is computed it is translated into
        the command of the slave robot
        """

        cmd = np.zeros(self._num_slave_jnts)

        scale = self.master_js_scale()

        if self._scale_from_home:

            for i, j in zip(range(self._num_master_jnts), range(self._num_slave_jnts)):

                cmd[j] += scale[i] * (self._slave_jlims[j]['upper'] - self._slave_jlims[j]['lower'])

        else:

            for i, j in zip(range(self._num_master_jnts), range(self._num_slave_jnts)):

                cmd[j] = self._slave_jlims[j]['lower'] + scale[i] * (self._slave_jlims[j]['upper'] - self._slave_jlims[j]['lower'])

        self._old_cmd = cmd

        return cmd, np.zeros(self._num_slave_jnts)


    def run(self):
        """
        ctrlr function,
        computes the scale and find the required velocity
        """

        rate = rospy.timer.Rate(self._rate)

        while not rospy.is_shutdown():

            goal_js_pos, goal_js_vel = self.compute_cmd()

            self._bullet_robot.exec_position_cmd(goal_js_pos)#(cmd=goal_js_pos)

            # self._bullet_robot.simple_step()
            
            rate.sleep()


