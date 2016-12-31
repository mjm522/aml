#!/usr/bin/env python

# A simply python controller for Baxter compatible with GPS (TODO)
import rospy
import cv2

import tf
from tf import TransformListener

import numpy as np
import quaternion

from aml_ctrl.controllers.osc_torque_controller import OSCTorqueController
from aml_ctrl.controllers.osc_postn_controller import OSCPositionController
from aml_ctrl.controllers.osc_moveit_baxter_controller import BaxterMoveItController
from aml_ctrl.utilities.min_jerk_interp import MinJerkInterp
from aml_ctrl.utilities.lin_interp import LinInterp
from config import BOX_TYPE_1
from aml_io.io import save_data, load_data
from aml_perception import camera_sensor
import aml_calib
import camera_calib
import random


def set_reset_jnt_pos():
    #use cuff buttons to set the reset positions
    #call this function to set the values
    limb_l = 'left'
    limb_r = 'right'
    arm_l = BaxterArm(limb_l)
    arm_r = BaxterArm(limb_r)
    reset_jnt_pos = {}
    reset_jnt_pos[limb_l]  = arm_l._state['position']
    reset_jnt_pos[limb_r] = arm_r._state['position']
    np.save('reset_jnt_pos.npy', reset_jnt_pos)

def get_reset_jnt_pos():
    
    try:
        reset_jnt_pos = np.load('reset_jnt_pos.npy').item()
    except Exception as e:
        raise e

    return reset_jnt_pos

class CollectPokeData():

    def __init__(self, robot_interface, box_config=BOX_TYPE_1):
        self._robot = robot_interface

        # self.calib_extern_cam()

        self._interp_fn = MinJerkInterp(dt=0.5, tau=15.)
        # self._interp_fn = LinInterp(dt=0.5, tau=15.)
        self._box_tf = TransformListener()

        # self._ctrlr    = BaxterMoveItController()
        # self._ctrlr  = OSCTorqueController(robot_interface)
        self._ctrlr  = OSCPositionController(robot_interface)
        self._ctrlr.set_active(True)

        self._goal_pos_old = None
        self._goal_ori_old = None
        self._goal_pos_new = None
        self._goal_ori_new = None
        self._interp_traj  = None

        self._tfmn_time    = None

        self._box_length  = box_config['length']
        self._box_breadth = box_config['breadth']
        self._box_height  = box_config['height']
        self._reset_jnt_pos = get_reset_jnt_pos()

        #this is the option to choose to opt between various locations on the box
        self.update_choice()

    def calib_extern_cam(self):
        hand_eye_calib = camera_calib.BaxterEyeHandCalib()
        hand_eye_calib.self_calibrate()
        rospy.sleep(10)

    def reset_arm(self):
        #reset the arm under consideration to the position
        self._robot.exec_position_cmd(self._reset_jnt_pos[self._robot._limb])
        # self._robot.untuck_arm()

    def plan_traj(self):
        #minimum jerk trajectory for left arm
        self.reset_arm()
        robot_pos, robot_ori = self._robot.get_ee_pose()

        self._interp_fn.configure(robot_pos, robot_ori, self._goal_pos_new, self._goal_ori_new)
        self._interp_traj   = self._interp_fn.get_interpolated_trajectory()

    def rand_location_on_box(self, choice=1):
        #this function randomly chooses between four locations
        #on the box
        if choice == 1:
            vect = np.array([0.3,  -0.5, 0.])
        elif choice == 2:
            vect = np.array([-0.3, -0.5, 0.])
        elif choice == 3:
            vect = np.array([0.,   -0.5, -0.3])
        else:
            vect = np.array([0.,   -0.5, 0.3])

        return np.multiply(vect, np.array([self._box_length, self._box_height, self._box_breadth]))

    def compute_goal_pose(self):
        box_pos, box_ori = self.get_box_pose()
        goal_pos  = box_pos + np.dot(quaternion.as_rotation_matrix(box_ori), self.rand_location_on_box(self._choice))
        
        self._goal_pos_new = goal_pos
        self._goal_ori_new = box_ori

        if (self._goal_pos_old is None) and (self._goal_ori_old is None):
            self._goal_pos_old = goal_pos
            self._goal_ori_old = box_ori

    def check_change_goal_pose(self):
        self.compute_goal_pose()

        if (np.linalg.norm(self._goal_pos_new-self._goal_pos_old) < 0.05) and (self._interp_traj is not None):
            return False
        else:
            self._goal_pos_old = self._goal_pos_new
            self._goal_ori_old = self._goal_ori_new
            return True

    def get_box_pose(self):

        while self._tfmn_time is None:
            try:
                self._tfmn_time = self._box_tf.getLatestCommonTime('base', 'box')
            except tf.Exception:
                pass

        if self._tfmn_time is not None:
            box_pos, box_ori     = self._box_tf.lookupTransform('base', 'box', self._tfmn_time)

            self._tfmn_time = None

            box_pos = np.array([box_pos[0],box_pos[1],box_pos[2]])
            
            box_ori = np.quaternion(box_ori[3],box_ori[0],box_ori[1],box_ori[2])

        return box_pos, box_ori


    def update_traj(self):
        #update the traj only if there is change in goal position
        if self.check_change_goal_pose():
            self.plan_traj()
            return True
        else:
            return False

    def update_choice(self):
          self._choice = random.randint(1,4)


def main(robot_interface):

    cpd = CollectPokeData(robot_interface=robot_interface)
    
    n_steps = len(cpd._interp_fn.timesteps)

    finished = False
    t = 0

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():# and not finished:
  
        cpd.get_box_pose()

        if finished:
            #once the earlier rollout is done
            #choose a different location
            #on the box
            cpd.update_choice()

        #reset the index if there is a change in the trajectory
        if (cpd.update_traj()):
            t = 0
            finished = False
            print "Choice is \t", cpd._choice
            print "Detected change in position, trajectory updated, moving to neutral pose..."

        error_lin = np.linalg.norm(cpd._ctrlr._error['linear'])

        if not finished:
            
            goal_pos  = cpd._interp_traj['pos_traj'][t]
            goal_ori  = cpd._interp_traj['ori_traj'][t]
            goal_vel  = cpd._interp_traj['vel_traj'][t]
            goal_omg  = cpd._interp_traj['omg_traj'][t]

            print "Sending goal ",t, " goal_pos:", goal_pos.ravel()

            if np.any(np.isnan(goal_pos)):

                print "Goal", t, "is NaN, that is not good, we will skip it!"

            else:
                cpd._ctrlr.set_goal(goal_pos=goal_pos, 
                                    goal_ori=goal_ori, 
                                    goal_vel=goal_vel, 
                                    goal_omg=goal_omg, 
                                    orientation_ctrl = False)

                # print "Waiting..." 
                lin_error, ang_error, success, time_elapsed = cpd._ctrlr.wait_until_goal_reached(timeout=1.0)
                # print "lin_error: %0.4f ang_error: %0.4f elapsed_time: (secs,nsecs) = (%d,%d)"%(lin_error,ang_error,time_elapsed.secs,time_elapsed.nsecs), " reached: ", success

            t = (t+1)
            finished = (t == n_steps)

            rate.sleep()

    lin_error, ang_error, success, time_elapsed = cpd._ctrlr.wait_until_goal_reached(timeout=10)
    cpd._ctrlr.set_active(False)

def moveit_main(robot_interface):

    cpd = CollectPokeData(robot_interface=robot_interface)
    
    limb_group = robot_interface._limb_group

    cpd._ctrlr.set_group_handles(limb_group=limb_group)
    # cpd._ctrlr.self_test(limb_group=limb_group)
    # cpd._ctrlr.add_static_objects_to_scene(limb_group=limb_group, obj_pos=None, obj_ori=None)
    
    cpd.compute_goal_pose()
    
    plan = cpd._ctrlr.get_plan(limb_group=limb_group, pos=cpd._goal_pos_new, ori=cpd._goal_ori_new, wait_time=15)
    
    # cpd._ctrlr.execute_plan(limb_group=limb_group, plan=plan, real_robot=True)
    
    cpd._ctrlr.clean_shutdown()


if __name__ == "__main__":
    rospy.init_node('poke_box', anonymous=True)
    from aml_robot.baxter_robot import BaxterArm
    limb = 'left'
    arm = BaxterArm(limb)
    arm.untuck_arm()
    main(arm)
    # set_reset_jnt_pos()
    # moveit_main(arm)
    

    

