#!/usr/bin/env python

# A simply python controller for Baxter compatible with GPS (TODO)


import rospy
import cv2

import tf
from tf import TransformListener

import numpy as np

from aml_ctrl.controllers.osc_torque_controller import OSCTorqueController
from aml_ctrl.controllers.osc_postn_controller import OSCPositionController
from aml_io.io import save_data, load_data
from aml_perception import camera_sensor

def main(robot_interface):

    box_tf = TransformListener()
    ctrlr  = OSCPositionController(robot_interface)
    ctrlr.set_active(True)

    box_length  = 0.210 #m
    box_breadth = 0.153
    box_height  = 0.080

    reach_thr = 0.12

    arm_pos, arm_ori = robot_interface.get_ee_pose()

    while not rospy.is_shutdown():

        try:
            tfmn_time = box_tf.getLatestCommonTime('base', 'box')
            flag_box = True
        except tf.Exception:
            print "Some exception occured while getting the transformation!!!"

        if flag_box:
            flag_box = False
            box_pos, box_ori     = box_tf.lookupTransform('base', 'box', tfmn_time)
            box_pos = np.array([box_pos[0],box_pos[1],box_pos[2]])
            box_ori = np.quaternion(box_ori[3],box_ori[0],box_ori[1],box_ori[2])

            goal_pos  = box_pos + np.dot(quaternion.as_rotation_matrix(box_ori), np.array([box_length/2., -box_height/2.,0.]))


            print "Sending goal ",t, " goal_pos:",goal_pos.ravel()

            if np.any(np.isnan(goal_pos)):
                print "Goal", t, "is NaN, that is not good, we will skip it!"
            else:
                ctrlr.set_goal(goal_pos, arm_ori)

                print "Waiting..." 
                lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=1.0)
                print "lin_error: %0.4f ang_error: %0.4f elapsed_time: (secs,nsecs) = (%d,%d)"%(lin_error,ang_error,time_elapsed.secs,time_elapsed.nsecs), " reached: ", success

            rate.sleep()


        if lin_error < reach_thr:
            break

    # save_data(data,'data_std1.pkl')



if __name__ == "__main__":
    rospy.init_node('poke_box', anonymous=True)
    from aml_robot.baxter_robot import BaxterArm
    limb = 'right'
    arm = BaxterArm(limb)
    main(arm)


    

