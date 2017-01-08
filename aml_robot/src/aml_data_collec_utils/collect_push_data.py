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
from config import config
from aml_io.io import save_data, load_data
from aml_perception import camera_sensor
import aml_calib
import camera_calib
import random
import cv2

from ros_transform_utils import get_pose, transform_to_pq, pq_to_transform


class BoxObject(object):

    def __init__(self):

        self._tf              = TransformListener()

        self._dimensions      = np.array([config['box_type']['length'], config['box_type']['height'], config['box_type']['breadth']])

        self._frame_name      = 'box'

        self._base_frame_name = 'base'


        self._box_reset_pos0 = None


        # Publish
        self._br = tf.TransformBroadcaster()

        update_rate = 30.0
        update_period = rospy.Duration(1.0/update_rate)
        rospy.Timer(update_period, self.update_frames)

    def get_pose(self):

        time = self._tf.getLatestCommonTime(self._base_frame_name, self._frame_name)

        return get_pose(self._tf,self._base_frame_name,self._frame_name, time)

    def get_pre_push(self,idx):

        time = self._tf.getLatestCommonTime(self._base_frame_name, self._frame_name)

        return get_pose(self._tf,self._base_frame_name,'pre_push%d'%(idx,), time)

    def get_push_pose(self):

        time = self._tf.getLatestCommonTime(self._base_frame_name, self._frame_name)

        return get_pose(self._tf,self._base_frame_name,'pre_push%d'%(idx,), time)

    def get_reset_pose(self):

        time = self._tf.getLatestCommonTime(self._base_frame_name, self._frame_name)

        pose = get_pose(self._tf, self._base_frame_name,self._frame_name, time)

        p, q = transform_to_pq(pose)

        reset_offset = config['reset_spot_offset']
        pos_rel_box = np.array([reset_offset[0],reset_offset[1],reset_offset[2],1])
        pos = np.asarray(np.dot(pose,pos_rel_box)).ravel()[:3]

        return pos, q

    def update_frames(self,event):

        try:

            poses, _ = self.get_pushes()
            count = 0
            for pose in poses:

                now = rospy.Time.now()

                self._br.sendTransform(pose['pos'], pose['ori'], now, pose['name'], 'base')


        except Exception as e:
            print e
            pass

    # Computes a list of "pushes", a push contains a pre-push pose, 
    # a push action (goal position a push starting from a pre-push pose) 
    # and its respective name
    # It also returns the current box pose, and special reset_push
    def get_pushes(self):

        success = False
        max_trials = 200
        trial_count = 0

        pos = pose = time = None

        p = q = None

        while trial_count < max_trials and not success:

            try:
                time = self._tf.getLatestCommonTime(self._base_frame_name, self._frame_name)

                pose = get_pose(self._tf, self._base_frame_name,self._frame_name, time)
                p, q = transform_to_pq(pose)

                reset_pos, reset_q = self.get_reset_pose()

                success = True
            except Exception as e:
                trial_count += 1


        
        if success:
            pre_push_offset = config['pre_push_offsets']
            positions = np.array([[pre_push_offset[0], pre_push_offset[1], 0,                  1],
                                  [0                 , pre_push_offset[1], pre_push_offset[2], 1]])
            signs = [1,-1]


            pushes = []
            count = 0
            for position in positions:
                for s in signs:

                    # position relative to the box        
                    pos_rel_box = np.multiply(position,np.array([s,1,s,1]))
                    pos = np.asarray(np.dot(pose,pos_rel_box)).ravel()[:3]

                    push_action = np.asarray(np.dot(pose,np.array([0, pre_push_offset[1], 0, 1]))).ravel()[:3]
                    pushes.append({'pos': pos, 'ori': q, 'push_action': push_action, 'name' : 'pre_push%d'%(count,)})

                    count += 1


            if self._box_reset_pos0 is None:
                self._box_reset_pos0 = reset_pos

            reset_push = {'pos': reset_pos, 'ori': reset_q, 'push_action': self._box_reset_pos0 - reset_pos, 'name' : 'reset_spot'}

            pushes.append(reset_push)

            return pushes, pose
        else:
            return [], None


# class Sample(object):

#     def __init__(self,sample_id):

#         self._id = sample_id



class PushMachine(object):

    def __init__(self,robot_interface):

        self._push_counter = 0
        self._box = BoxObject()
        self._robot = robot_interface


    def run(self):

        push_finished = True
        pre_push_finished = True

        t = 0

        rate = rospy.Rate(10)

        idx = 0
        while not rospy.is_shutdown():# and not finished:

            pushes = None
            box_pose = None
            
            pushes, box_pose = self._box.get_pushes()

            if pushes:

                print "Moving to neutral position ..."
                    
                self._robot.untuck_arm()

                print "Moving to pre-push position ..."
                    
                self.goto_pose(goal_pos=pushes[idx]['pos'], goal_ori=None)

                print "Gonna push the box ..."

                self.goto_pose(goal_pos=pushes[idx]['push_action'], goal_ori=None)
                self._push_counter += 1

                idx = (idx+1)%(len(pushes)-1)

                if idx == 0:
                    self.reset_box(pushes[-1])


            rate.sleep()

    def goto_pose(self,goal_pos, goal_ori): 

        start_pos, start_ori = self._robot.get_ee_pose()

        if goal_ori is None:
             goal_ori = start_ori

        goal_ori = quaternion.as_float_array(goal_ori)[0]
        success, js_pos = self._robot.ik(goal_pos,goal_ori)

        if success:
            self._robot.move_to_joint_position(js_pos)
        else:
            print "Couldnt find a solution"

    def reset_box(self,reset_push):
        print "RESET PUSH", reset_push




if __name__ == "__main__":
    rospy.init_node('poke_box', anonymous=True)
    from aml_robot.baxter_robot import BaxterArm
    limb = 'left'
    arm = BaxterArm(limb)
    push_machine = PushMachine(arm)

    print "calling run"
    push_machine.run()
   
    

    

