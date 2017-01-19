#!/usr/bin/env python

# A simply python controller for Baxter compatible with GPS (TODO)
import rospy
import cv2

import tf
from tf import TransformListener

import numpy as np
import quaternion

from aml_ctrl.controllers.os_controllers.os_torque_controller import OSTorqueController
from aml_ctrl.controllers.os_controllers.os_postn_controller import OSPositionController
from aml_ctrl.controllers.os_controllers.os_postn_controller import OSPositionController
from aml_ctrl.controllers.os_controllers.os_jt_torque_controller import OSJTTorqueController
from aml_ctrl.utilities.min_jerk_interp import MinJerkInterp
from aml_ctrl.traj_generator.js_traj_generator import JSTrajGenerator
from aml_ctrl.utilities.lin_interp import LinInterp
from config import BOX_TYPE_1
from aml_io.io_tools import save_data, load_data
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

        self.calib_extern_cam()
        print asfdkla

        self._interp_fn = MinJerkInterp(dt=0.5, tau=15.)
        # self._interp_fn = LinInterp(dt=0.5, tau=15.)

        self._box_tf = TransformListener()


        self._goal_pos_old = None
        self._goal_ori_old = None
        self._goal_pos_new = None
        self._goal_ori_new = None

        self._goal_pre_push_pos_new = None
        self._goal_pre_push_ori_new = None

        self._interp_traj_push  = None
        self._interp_traj_pre_push  = None

        self._tfmn_time    = None

        self._reset_jnt_pos = get_reset_jnt_pos()

    def calib_extern_cam(self):
        hand_eye_calib = camera_calib.BaxterEyeHandCalib()
        hand_eye_calib.self_calibrate()
        rospy.sleep(10)

    def get_box_pose(self):

        while self._tfmn_time is None:
            try:
                self._tfmn_time = self._box_tf.getLatestCommonTime('base', 'box')
            except tf.Exception:
                pass

        if self._tfmn_time is not None:
            box_pos, box_ori   = self._box_tf.lookupTransform('base', 'box', self._tfmn_time)

            self._tfmn_time = None

            box_pos = np.array([box_pos[0],box_pos[1],box_pos[2]])
            
            box_ori = np.quaternion(box_ori[3],box_ori[0],box_ori[1],box_ori[2])

        return box_pos, box_ori

    def get_tip_pose(self):
        box_pos, box_ori = self.get_box_pose()
        
        ee_pos,  ee_ori = self._robot.get_ee_pose()

        tip_pos1 = box_pos + np.array([0., 0.0, 0.096])
        
        tip_ori = ee_ori

        rel_tip_pos = tip_pos1 - ee_pos

        raw_input('press any key')

        ee_pos,  ee_ori = self._robot.get_ee_pose()

        tip_pos2 = ee_pos + rel_tip_pos

        rel_dist = tip_pos2-tip_pos1

        print "the cup is this far from marker, in base frame"

        print rel_dist

        print "the cup is this far from marker, in box frame"

        print np.dot(quaternion.as_rotation_matrix(box_ori).T, rel_dist)

        print "bye..."


def execute_trajectory(robot_interface, trajectory, rate):

    ctrlr  = OSPositionController(robot_interface)
    # ctrlr    = BaxterMoveItController()
    # ctrlr  = OSTorqueController(robot_interface)
    ctrlr.set_active(True)

    t = 0
    n_steps = len(trajectory['pos_traj'])

    while t < n_steps:
    
        goal_pos  = trajectory['pos_traj'][t]
        goal_ori  = trajectory['ori_traj'][t]
        goal_vel  = trajectory['vel_traj'][t]
        goal_omg  = trajectory['omg_traj'][t]

        print "Sending goal ",t, " goal_pos:", goal_pos.ravel()

        if np.any(np.isnan(goal_pos)):

            print "Goal", t, "is NaN, that is not good, we will skip it!"

        else:
            
            ctrlr.set_goal(goal_pos=goal_pos, 
                           goal_ori=goal_ori, 
                           goal_vel=goal_vel, 
                           goal_omg=goal_omg, 
                           orientation_ctrl = True)

            # print "Waiting..." 
            lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=1.0)
            # print "lin_error: %0.4f ang_error: %0.4f elapsed_time: (secs,nsecs) = (%d,%d)"%(lin_error,ang_error,time_elapsed.secs,time_elapsed.nsecs), " reached: ", success

        t = (t+1)

        rate.sleep()

        finished = (t == n_steps)

    lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=10)
    ctrlr.set_active(False)

    return finished

def reach_point(robot_interface, goal_pos, goal_ori):

    interp_fn = MinJerkInterp()

    rate = rospy.Rate(10)

    robot_pos, robot_ori = robot_interface.get_ee_pose()

    if goal_ori is None:
        goal_ori = robot_ori
    
    interp_fn.configure(robot_pos, robot_ori, goal_pos, goal_ori)

    interp_traj  = interp_fn.get_interpolated_trajectory()

    execute_trajectory(robot_interface=robot_interface, trajectory=interp_traj, rate=rate)


def execute_trajectory_jt(robot_interface, goal_pos, goal_ori):
    rate = rospy.Rate(10)

    ctrlr  = OSJTTorqueController(robot_interface)

    ctrlr.set_active(True)

    robot_pos, robot_ori = robot_interface.get_ee_pose()

    if goal_ori is None:
        goal_ori = robot_ori

    error = np.linalg.norm(goal_pos-robot_pos)

    while error > 0.01:

        ctrlr.set_goal(goal_pos=goal_pos, 
                       goal_ori=goal_ori, 
                       orientation_ctrl = True)

        lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=1.0)

        rate.sleep()

        error = np.linalg.norm(goal_pos-robot_pos)

    lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=10)
    ctrlr.set_active(False)

def execute_trajectory_js(robot_interface, goal_pos, goal_ori):
    
    rate = rospy.Rate(5)

    start_pos, start_ori = robot_interface.get_ee_pose()

    if goal_ori is None:
         goal_ori = start_ori

    goal_ori = quaternion.as_float_array(goal_ori)[0]
    success, js_pos = robot_interface.ik(goal_pos,goal_ori)

    if success:
        robot_interface.move_to_joint_position(js_pos)
    else:
        print "Couldnt find a solution"


def get_pre_push_pose(pre_push_pos):

    #in box frame offset from ee_pos if the tip is touching the box top
    box_top      = np.array([-0.01350, 0.16090282, -0.00500357])

    cup_top      = np.array([-0.09949958, 0.00233311, -0.00379943])

    box_dim      = np.array([BOX_TYPE_1['length'], BOX_TYPE_1['height'], BOX_TYPE_1['breadth']])

    if pre_push_pos==0:

        return box_top

    elif pre_push_pos==1:

        return box_top + np.multiply(np.array([0., -0.9, -0.9]), box_dim)

    elif pre_push_pos==2:

        return box_top + np.multiply(np.array([-0.9, -0.9, 0.]), box_dim)

    elif pre_push_pos==4:

        return box_top + np.multiply(np.array([0.3, -0.9, 0.]), box_dim)

    #for resetting the box position
    elif pre_push_pos==5:

        return box_top + np.multiply(np.array([0.0, 1.5, 0.]), box_dim) + cup_top

    #for going inside the cup
    elif pre_push_pos==6:

        return box_top + cup_top 


def get_push_pose(push_side):

    if push_side==1:

        return get_pre_push_pose(1) + np.array([0., 0., 0.10])

    elif push_side==2:

        return get_pre_push_pose(2) + np.array([0.10, 0., 0.])

    elif push_side==4:

        return get_pre_push_pose(4) + np.array([-0.10, 0., 0.])

    elif push_side==6:

        return get_pre_push_pose(6) + get_pre_push_pose(6) - get_pre_push_pose(0) 


def reach_pre_push_pose(object_interface, robot_interface, pre_push_pos=0):

    box_pos, box_ori = object_interface.get_box_pose()

    if pre_push_pos==0:

        print "Chosen side : top"

        goal_pos = box_pos + np.dot(quaternion.as_rotation_matrix(box_ori), get_pre_push_pose(pre_push_pos))
        goal_ori = None

    elif pre_push_pos==1:

        print "Chosen side : A"

        goal_pos = box_pos + np.dot(quaternion.as_rotation_matrix(box_ori), get_pre_push_pose(pre_push_pos))
        goal_ori = None

    elif pre_push_pos==2:

        print "Chosen side : B"
        
        goal_pos = box_pos + np.dot(quaternion.as_rotation_matrix(box_ori), get_pre_push_pose(pre_push_pos))
        goal_ori = None

    elif pre_push_pos==4:

        print "Chosen side : D"

        goal_pos = box_pos + np.dot(quaternion.as_rotation_matrix(box_ori), get_pre_push_pose(pre_push_pos))
        goal_ori = None

    elif pre_push_pos==5:

        print "Reset chosen ..."

        goal_pos = box_pos + np.dot(quaternion.as_rotation_matrix(box_ori), get_pre_push_pose(pre_push_pos))
        goal_ori = None

    elif pre_push_pos==6:

        print "Moving inside the cup ..."

        goal_pos = box_pos + np.dot(quaternion.as_rotation_matrix(box_ori), get_pre_push_pose(pre_push_pos))
        goal_ori = None

    # reach_point(robot_interface=robot_interface, goal_pos=goal_pos, goal_ori=goal_ori)
    # execute_trajectory_jt(robot_interface=robot_interface, goal_pos=goal_pos, goal_ori=goal_ori)
    execute_trajectory_js(robot_interface=robot_interface, goal_pos=goal_pos, goal_ori=goal_ori)

def push_box(object_interface, robot_interface, push_side=1):

    box_pos, box_ori = object_interface.get_box_pose()

    if push_side==1:
        #push side A

        goal_pos = box_pos + np.dot(quaternion.as_rotation_matrix(box_ori), get_push_pose(push_side))
        goal_ori = None

    elif push_side==2:
        #push side B

        goal_pos = box_pos + np.dot(quaternion.as_rotation_matrix(box_ori), get_push_pose(push_side))
        goal_ori = None

    elif push_side==4:
        #push side D

        goal_pos = box_pos + np.dot(quaternion.as_rotation_matrix(box_ori), get_push_pose(push_side))
        goal_ori = None

    #this is six since there is not pre-push 5, 
    elif push_side==6:

        goal_pos = box_pos + np.dot(quaternion.as_rotation_matrix(box_ori), get_push_pose(push_side))
        goal_ori = None

    # reach_point(robot_interface=robot_interface, goal_pos=goal_pos, goal_ori=goal_ori)
    # execute_trajectory_jt(robot_interface=robot_interface, goal_pos=goal_pos, goal_ori=goal_ori)
    execute_trajectory_js(robot_interface=robot_interface, goal_pos=goal_pos, goal_ori=goal_ori)


def reset_the_box(object_interface, robot_interface):

    robot_interface.untuck_arm()

    reach_pre_push_pose(object_interface=object_interface, robot_interface=robot_interface, pre_push_pos=5)
    reach_pre_push_pose(object_interface=object_interface, robot_interface=robot_interface, pre_push_pos=6)
    push_box(object_interface=object_interface, robot_interface=robot_interface, push_side=6)


def main(robot_interface):

    cpd = CollectPokeData(robot_interface=robot_interface)
    
    n_steps = len(cpd._interp_fn.timesteps)

    push_finished = True
    pre_push_finished = True

    t = 0

    rate = rospy.Rate(10)

    side_list = [1,2,4]

    while not rospy.is_shutdown():# and not finished:

        side = side_list[random.randint(0,2)]

        print "Moving to neutral position ..."
        
        robot_interface.untuck_arm()

        print "Moving to pre-push position ..."
        
        reach_pre_push_pose(object_interface=cpd, robot_interface=robot_interface, pre_push_pos=side)

        print "Gonna push the box ..."

        push_box(object_interface=cpd,robot_interface=robot_interface, push_side=side)


def debug_main(robot_interface):

    cpd = CollectPokeData(robot_interface=robot_interface)

    reset_the_box(object_interface=cpd, robot_interface=robot_interface)


def find_box_stuff(robot_interface):

    cpd = CollectPokeData(robot_interface=robot_interface)
    
    cpd.get_tip_pose()


if __name__ == "__main__":
    rospy.init_node('poke_box', anonymous=True)
    from aml_robot.baxter_robot import BaxterArm
    limb = 'left'
    arm = BaxterArm(limb)
    # main(arm)
    debug_main(arm)

    # find_box_stuff(robot_interface=arm)

    # main(arm)

    

    
