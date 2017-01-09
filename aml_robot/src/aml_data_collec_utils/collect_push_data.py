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
from record_sample import RecordSample
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
        tip_offset = config['end_effector_tip_offset']
        pos_rel_box = np.array([reset_offset[0],reset_offset[1]+tip_offset[1],reset_offset[2],1])
        pos = np.asarray(np.dot(pose,pos_rel_box)).ravel()[:3]

        return pos, q

    def update_frames(self,event):

        try:

            pushes, _, _ = self.get_pushes()
            count = 0
            for push in pushes:

                now = rospy.Time.now()
                for pose in push['poses']:
                    self._br.sendTransform(pose['pos'], pose['ori'], now, push['name'], 'base')


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

        pos = pose = time = ee_pose = None

        box_pos = q = None

        while trial_count < max_trials and not success:

            try:
                time = self._tf.getLatestCommonTime(self._base_frame_name, self._frame_name)

                # Getting box center (adding center offset to retrieved pose)
                pose = get_pose(self._tf, self._base_frame_name,self._frame_name, time)
                box_center_offset = config['box_center_offset']
                box_pos = np.asarray(np.dot(pose,np.array([box_center_offset[0],box_center_offset[1],box_center_offset[2],1]))).ravel()[:3]
                _, q = transform_to_pq(pose)
                pose = pq_to_transform(self._tf,box_pos,q)

                time = self._tf.getLatestCommonTime(self._base_frame_name, 'left_gripper')
                ee_pose = get_pose(self._tf, self._base_frame_name,'left_gripper', time)

                ee_pos, q = transform_to_pq(ee_pose)

                reset_pos, reset_q = self.get_reset_pose()

                success = True
            except Exception as e:
                print e
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
                    pushes.append({'poses': [{'pos': pos, 'ori': q}], 'push_action': push_action, 'name' : 'pre_push%d'%(count,)})

                    count += 1


            if self._box_reset_pos0 is None:
                self._box_reset_pos0 = reset_pos

            # Reset push is a special kind of push
            
            reset_offset = config['reset_spot_offset']
            pre_reset_offset = config['pre_reset_offsets']
            pos_rel_box = np.array([reset_offset[0],reset_offset[1]+pre_reset_offset[1],reset_offset[2],1])
            pre_reset_pos = np.asarray(np.dot(pose,pos_rel_box)).ravel()[:3]

            reset_displacement = (self._box_reset_pos0 - reset_pos)
            reset_push = {'poses': [{'pos': pre_reset_pos, 'ori': reset_q}, {'pos': reset_pos, 'ori': reset_q}], 'push_action': reset_displacement, 'name' : 'reset_spot'}
            

            # pushes.append(reset_push)

            return pushes, pose, reset_push
        else:
            return [], None, None


# class Sample(object):

#     def __init__(self,sample_id):

#         self._id = sample_id



class PushMachine(object):

    def __init__(self,robot_interface):

        self._push_counter = 0
        self._box = BoxObject()
        self._robot = robot_interface

        self._states = {'RESET': 0, 'PUSH' : 1}
        self._state = self._states['RESET']

        self._record_sample = RecordSample(robot_interface=robot_interface, record_rate=50)


    def compute_next_state(self,idx):

        # Decide next state
        if idx == 0 and self._push_counter > 0 and self._state != self._states['RESET']:
            self._state = self._states['RESET']
        else:
            self._state = self._states['PUSH']

    def goto_next_state(self,idx,pushes, box_pose, reset_push):

        success = True


        # Take machine to next state
        if self._state == self._states['RESET']:
            print "RESETING WITH NEW POSE"

            self._robot.untuck_arm()

            success = self.reset_box(reset_push)

        elif self._state == self._states['PUSH']:
            print "Moving to neutral position ..."
                        
            self._robot.untuck_arm()

            print "Moving to pre-push position ..."
                    
            # There might be a sequence of positions prior to a push action
            
            for goal in pushes[idx]['poses']:
                success = success and self.goto_pose(goal_pos=goal['pos'], goal_ori=None)


            if success:
                pass
                # self._record_sample.start_record(idx)

            print "Gonna push the box ..."

            success = success and self.goto_pose(goal_pos=pushes[idx]['push_action'], goal_ori=None)
            
            if success:
                self._push_counter += 1
            
                # self._record_sample.stop_record(success)
            
            idx = (idx+1)%(len(pushes))
        else:
            print "UNKNOWN STATE"


        return idx, success



    def run(self):

        push_finished = True
        pre_push_finished = True

        t = 0

        rate = rospy.Rate(10)

        idx = 0
        while not rospy.is_shutdown():# and not finished:

            pushes = None
            box_pose = None
            
            pushes, box_pose, reset_push = self._box.get_pushes()

            if pushes:

                self.compute_next_state(idx)

                idx, success = self.goto_next_state(idx, pushes, box_pose, reset_push)

                


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

        return success

    def reset_box(self,reset_push):
        success = True
        # There might be a sequence of positions prior to a push action
        for goal in reset_push['poses']:
            print "going to:", goal
            success = success and self.goto_pose(goal_pos=goal['pos'], goal_ori=None)


        ee_pos, _ = self._robot.get_ee_pose()

        success = success and self.goto_pose(goal_pos=ee_pos+reset_push['push_action'], goal_ori=None)

        return success



if __name__ == "__main__":
    rospy.init_node('poke_box', anonymous=True)
    from aml_robot.baxter_robot import BaxterArm
    limb = 'left'
    arm = BaxterArm(limb)
    push_machine = PushMachine(arm)

    print "calling run"
    push_machine.run()
   
    

    

