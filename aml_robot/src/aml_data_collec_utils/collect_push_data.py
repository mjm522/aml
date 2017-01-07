#!/usr/bin/env python

# A simply python controller for Baxter compatible with GPS (TODO)
import rospy
import cv2

import tf
from tf import TransformListener

import numpy as np
import quaternion

from aml_ctrl.utilities.min_jerk_interp import MinJerkInterp
from aml_ctrl.traj_generator.js_traj_generator import JSTrajGenerator
from aml_ctrl.utilities.lin_interp import LinInterp
from config import BOX_TYPE_1
from aml_io.io import save_data, load_data
from aml_perception import camera_sensor
import aml_calib
import camera_calib
import random
import cv2


def get_pose(tf, target, source, time):
    """
    Utility function that uses tf to return the position of target
    relative to source at time
    tf: Object that implements TransformListener
    target: Valid label corresponding to target link
    source: Valid label corresponding to source link
    time: Time given in TF's time structure of secs and nsecs
    """
    
    # Calculate the quaternion data for the relative position
    # between the target and source.
    translation, rot = tf.lookupTransform(target, source, time)

    #translation2, rot2 = tf.lookupTransform(source, target, time)

    # Get rotation and translation matrix from the quaternion data.
    # The top left 3x3 section is a rotation matrix.
    # The far right column is a translation vector with 1 at the bottom.
    # The bottom row is [0 0 0 1].
    transform = np.asmatrix(tf.fromTranslationRotation(translation, rot))

    return transform


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
        # print asfdkla

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

        tip_pos = box_pos + np.array([0., 0.0, 0.096])
        tip_ori = ee_ori

        rel_tip_pos = tip_pos - ee_pos

        np.save('rel_tip_pos.npy', rel_tip_pos)

        print "ee_pos", ee_pos
        print "tip pose", tip_pos
        print "box_pos", box_pos
        print "diff pos", box_pos-ee_pos
        print "rel_tip_pose", rel_tip_pos

        #HACK, FIX THIS


def reach_point(robot_interface, goal_pos, goal_ori):

    interp_fn = MinJerkInterp()

    rate = rospy.Rate(10)

    robot_pos, robot_ori = robot_interface.get_ee_pose()

    if goal_ori is None:
        goal_ori = robot_ori
    
    interp_fn.configure(robot_pos, robot_ori, goal_pos, goal_ori)

    interp_traj  = interp_fn.get_interpolated_trajectory()

    execute_trajectory(robot_interface=robot_interface, trajectory=interp_traj, rate=rate)



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
    box_top      = np.array([-0.00556044, 0.15976646, -0.00500357])

    box_dim      = np.array([BOX_TYPE_1['length'], BOX_TYPE_1['height'], BOX_TYPE_1['breadth']])

    if pre_push_pos==0:

        return box_top

    elif pre_push_pos==1:

        return box_top + np.multiply(np.array([0., -0.9, -0.9]), box_dim)

    elif pre_push_pos==2:

        return box_top + np.multiply(np.array([-0.7, -0.9, 0.]), box_dim)

    elif pre_push_pos==4:

        return box_top + np.multiply(np.array([0.7, -0.9, 0.]), box_dim)


def get_push_pose(push_side):

    if push_side==1:

        return get_pre_push_pose(1) + np.array([0., 0., 0.10])

    elif push_side==2:

        return get_pre_push_pose(2) + np.array([0.10, 0., 0.])

    elif push_side==4:

        return get_pre_push_pose(4) + np.array([-0.10, 0., 0.])


def reach_pre_push_pose(object_interface, robot_interface, pre_push_pos=0):

    robot_interface.untuck_arm()

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

    # reach_point(robot_interface=robot_interface, goal_pos=goal_pos, goal_ori=goal_ori)
    # execute_trajectory_jt(robot_interface=robot_interface, goal_pos=goal_pos, goal_ori=goal_ori)
    execute_trajectory_js(robot_interface=robot_interface, goal_pos=goal_pos, goal_ori=goal_ori)


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

    reach_pre_push_pose(object_interface=cpd, robot_interface=robot_interface, pre_push_pos=0)
    # push_box(object_interface=cpd, robot_interface=robot_interface, push_side=1)


if __name__ == "__main__":
    rospy.init_node('poke_box', anonymous=True)
    from aml_robot.baxter_robot import BaxterArm
    limb = 'left'
    arm = BaxterArm(limb)
    # main(arm)
    # debug_main(arm)

    BASE_LINK      = 'base'
    WRIST_3_LINK   = 'left_gripper'

    BOX_FRAME      = 'box'

    tf_listener = TransformListener()
    

    pose_cup_rel = [[-0.18924516, 0.98188187, -0.00970913,-0.12472268],
                    [-0.00677042,-0.01119235,-0.99991444,0.16930722],
                    [-0.98190653,-0.18916324,0.00876585,0.01366145],
                    [ 0.,0.,0.,1.]]


    pose_box_old = [[ -1.27950670e-02,5.76225280e-04,9.99917974e-01, 7.10263181e-01],
                    [  9.99893192e-01,7.07135649e-03,1.27906748e-02,9.45637319e-02],
                    [ -7.06340615e-03,9.99974832e-01,-6.66642214e-04,-1.28699791e-01],
                    [  0.00000000e+00,0.00000000e+00,0.00000000e+00,1.00000000e+00]]


    pose_cup_rel_box = [[ 0.99985641, 0.00955653,-0.01399392,-0.02357728],
                        [-0.00991683,0.99961515,-0.02590741,0.11813402],
                        [ 0.01374095,0.02604246,0.9995664,0.0010812 ],
                        [ 0.,0.,0.,1.]]



    r = rospy.Rate(10)
    while not rospy.is_shutdown():

        time = rospy.Time.now()
        try:

            time = tf_listener.getLatestCommonTime(BASE_LINK, BOX_FRAME)

            pose_box = get_pose(tf_listener, BASE_LINK, BOX_FRAME, time)

            print "BOX:", pose_box[:3,3].T

            raw_input("Take to another pose and press any key")

            time = tf_listener.getLatestCommonTime(BASE_LINK, WRIST_3_LINK)

            pose_ee1 = get_pose(tf_listener, BASE_LINK, WRIST_3_LINK, time)

            print "EE1:", pose_ee1

            box_ee1 = np.linalg.inv(pose_ee1)*pose_box

            print "BOX_EE1:", box_ee1


            tip_position1 = pose_ee1[:3,3].T + np.array([0,0,0.16296534])

            print "TIP_POSITION_1", tip_position1

            raw_input("Take to another pose and press any key")

            rospy.sleep(5)

            time = tf_listener.getLatestCommonTime(BASE_LINK, WRIST_3_LINK)
            pose_ee2 = get_pose(tf_listener, BASE_LINK, WRIST_3_LINK, time)

            tip_position2 = pose_ee2[:3,3].T + np.array([0,0,0.16296534]) 

            print "TIP_POSITION_1", tip_position2

            box_ee1 = np.linalg.inv(pose_ee1)*pose_box
            box_ee2 = np.linalg.inv(pose_ee2)*pose_box

            

            pose_ee12 = np.linalg.inv(pose_ee1)*pose_ee2


            pose_cup = pose_box*np.linalg.inv(pose_ee12)

            print "CUP:", pose_cup[:3,3].T
            print "CUP_REL:", pose_ee12.T



            #             p_1_0 = inv(p0)*p1
            # p_2_0 = inv(p0)*p2

            # p_1_2 = *p_1_0
        except tf.Exception:
            print "Fail"
            pass
        
        r.sleep()


    # main(arm)

    

    

