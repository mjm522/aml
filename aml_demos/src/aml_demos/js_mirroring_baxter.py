#!/usr/bin/env python

import rospy
import numpy as np
from aml_robot.baxter_robot import BaxterArm
from aml_ctrl.controllers.js_controllers.js_postn_controller import JSPositionController 
from aml_ctrl.controllers.js_controllers.js_torque_controller import JSTorqueController
from aml_ctrl.controllers.js_controllers.js_velocity_controller import JSVelocityController

Controller = JSPositionController

class JSMirroringBaxter():
    """
    This class implements a mirroring of the baxter robot
    This could also be used for other robots as well
    """

    def __init__(self, master_limb='left', slave_limb='right'):

        self._arm_master = BaxterArm(master_limb)
        self._arm_slave  = BaxterArm(slave_limb)

        self._arm_master.untuck()
        self._arm_slave.untuck()

        self._master_start_jpos  =  self._arm_master.state()['position']
        self._slave_start_jpos   =  self._arm_slave.state()['position']

        self._master_jlims = self._arm_master._jnt_limits

        self._slave_jlims = self._arm_slave._jnt_limits

        self._ctrlr_slave = Controller(self._arm_slave)

        self._scale_from_home = False

        self._num_master_jnts = 7

        self._num_slave_jnts = 7
        
        self._rate = 500 #Hz

        self._arm_slave.set_sampling_rate(500)

        self._arm_master.set_sampling_rate(500)
        
        self._ctrlr_slave.set_active(True)


    def master_js_scale(self, scale_from_home=False):

        scale = np.zeros(self._num_master_jnts)

        curr_js = self._arm_master.state()['position']

        if scale_from_home:

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

        cmd = self._arm_slave._untuck

        scale = self.master_js_scale(self._scale_from_home)

        if self._scale_from_home:

            for i, j in zip(range(self._num_master_jnts), range(self._num_slave_jnts)):

                cmd[j] += scale[i] * (self._slave_jlims[j]['upper'] - self._slave_jlims[j]['lower'])

        else:

            for i, j in zip(range(self._num_master_jnts), range(self._num_slave_jnts)):

                cmd[j] = self._slave_jlims[j]['lower'] + scale[i] * (self._slave_jlims[j]['upper'] - self._slave_jlims[j]['lower'])

        self._old_cmd = cmd

        return cmd, np.zeros(self._num_slave_jnts)


    def run(self):

        rate = rospy.timer.Rate(self._rate)

        while not rospy.is_shutdown():

            goal_js_pos, goal_js_vel = self.compute_cmd()

            self._ctrlr_slave.set_goal(goal_js_pos=goal_js_pos, 
                                       goal_js_vel=goal_js_vel, 
                                       goal_js_acc=np.zeros(self._num_slave_jnts))

            js_pos_error, success, time_elapsed = self._ctrlr_slave.wait_until_goal_reached(timeout=5.0)

            # print("lin_error: %0.4f ang_error: %0.4f elapsed_time: (secs,nsecs) = (%d,%d)"%(lin_error,ang_error,time_elapsed.secs,time_elapsed.nsecs), " success: ", success)

            #print "the command \n", cmd_right
            rate.sleep()


def main():

    rospy.init_node("js_mirror_demo_node")

    js_mirror = JSMirroringBaxter()

    js_mirror.run()


if __name__ == "__main__":
    main()
