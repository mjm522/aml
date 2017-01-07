import tf
from tf import TransformListener
import numpy as np
import quaternion
import rospy
from aml_ctrl.utilities.min_jerk_interp import MinJerkInterp
from aml_ctrl.controllers.os_controllers.os_bi_arm_controller import OSBiArmController

def test_reach_both_sides_box(right_arm, left_arm):
    flag_box = False
    box_tf = TransformListener()

    ctrlr  = OSBiArmController(right_arm=right_arm , left_arm=left_arm, mode='torque')

    min_jerk_interp = MinJerkInterp()

    left_task_complete = False
    right_task_complete = False
    box_length  = 0.410 #m
    box_breadth = 0.305
    box_height  = 0.107
    #for better show off ;)
    baxter_ctrlr.set_neutral()

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

            left_pos,  left_ori  = ctrlr.left_arm.get_ee_pose()
    
            right_pos, right_ori = ctrlr.right_arm.get_ee_pose()

            left_goal_pos  = box_pos + np.dot(quaternion.as_rotation_matrix(box_ori), np.array([box_length/2., -box_height/2.,0.]))
            right_goal_pos = box_pos + np.dot(quaternion.as_rotation_matrix(box_ori), np.array([-box_length/2.,-box_height/2.,0.]))
        
            error_left = np.linalg.norm(left_pos-left_goal_pos)
            error_right = np.linalg.norm(right_pos-right_goal_pos)
           
            #minimum jerk trajectory for left arm
            min_jerk_interp.configure(left_pos, left_ori, left_goal_pos, box_ori)
            min_jerk_traj_left   = min_jerk_interp.get_interpolated_trajectory()
            #minimum jerk trajectory for right arm
            min_jerk_interp.configure(right_pos, right_ori, right_goal_pos, box_ori)
            min_jerk_traj_right  = min_jerk_interp.get_interpolated_trajectory()
            
            for t in range(baxter_ctrlr.timesteps):
                #compute the left joint torque command
                ctrlr._left_arm_ctrlr.compute_cmd(goal_pos=min_jerk_traj_left['pos_traj'][t,:], 
                                                  goal_ori=min_jerk_traj_left['ori_traj'][t],  
                                                  orientation_ctrl=True)
                #compute the right joint torque command if the error is above some threshold
                ctrlr._right_arm_ctrlr.compute_cmd(goal_pos=min_jerk_traj_right['pos_traj'][t,:], 
                                                   goal_ori=min_jerk_traj_right['ori_traj'][t], 
                                                   orientation_ctrl=True)
                ctrlr._left_arm_ctrlr.send_cmd()
                ctrlr._right_arm_ctrlr.send_cmd()
            
            break

if __name__ == '__main__':
    rospy.init_node('reach_the_box')
    from aml_robot.baxter_robot import BaxterArm
    test_reach_both_sides_box(BaxterArm('right'), BaxterArm('left'))

        