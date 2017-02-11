#!/usr/bin/env python

# A simply python controller for Baxter compatible with GPS (TODO)
import tf
import os
import rospy
import argparse

import numpy as np
import random
import quaternion

import aml_calib
import camera_calib

from config import config
from tf import TransformListener
from record_sample import RecordSample
from aml_perception import camera_sensor
from aml_io.io_tools import save_data, load_data
from ros_transform_utils import get_pose, transform_to_pq, pq_to_transform

class BoxObject(object):

    def __init__(self):

        self._tf              = TransformListener()

        self._dimensions      = np.array([config['box_type']['length'], config['box_type']['height'], config['box_type']['breadth']])

        self._frame_name      = 'box'

        self._base_frame_name = 'base'


        self._box_reset_pos0 = None

        self._last_pushes = None


        # Publish
        self._br = tf.TransformBroadcaster()

        update_rate = 30.0
        update_period = rospy.Duration(1.0/update_rate)
        rospy.Timer(update_period, self.update_frames)

    #this is a util that makes the data in storing form
    def get_effect(self):
        pose, _, _   = self.get_pose()
        p, q   = transform_to_pq(pose)
        status = {}
        status['pos'] = p
        #all the files in package follows np.quaternion convention, that is
        # w,x,y,z while ros follows x,y,z,w convention
        status['ori'] = np.array([q[3],q[0],q[1],q[2]])

        return status


    def get_pre_push(self,idx):

        time = self._tf.getLatestCommonTime(self._base_frame_name, self._frame_name)

        return get_pose(self._tf,self._base_frame_name,'pre_push%d'%(idx,), time)

    def get_push_pose(self):

        time = self._tf.getLatestCommonTime(self._base_frame_name, self._frame_name)

        return get_pose(self._tf,self._base_frame_name,'pre_push%d'%(idx,), time)

    def get_reset_pose(self):

        time = self._tf.getLatestCommonTime(self._base_frame_name, self._frame_name)

        # Box pose
        pose, _, _ = self.get_pose()

        p, q = transform_to_pq(pose)

        reset_offset = config['reset_spot_offset']
        tip_offset = config['end_effector_tip_offset']
        pos_rel_box = np.array([reset_offset[0],reset_offset[1]+tip_offset[1],reset_offset[2],1])
        pos = np.asarray(np.dot(pose,pos_rel_box)).ravel()[:3]

        return pos, q

    def update_frames(self,event):

        try:

            if self._last_pushes is not None:
                pushes = self._last_pushes
                count = 0
                for push in pushes:

                    now = rospy.Time.now()
                    for pose in push['poses']:

                        self._br.sendTransform(pose['pos'], pose['ori'], now, "%s%d"%(push['name'],count), 'base')
                        count += 1


        except Exception as e:
            print "Error on update frames", e
            pass

    def get_pose(self, time = None):

        if time is None:
            time = self._tf.getLatestCommonTime(self._base_frame_name, self._frame_name)

        box_pos = box_q = None
        # Getting box center (adding center offset to retrieved pose)
        pose = get_pose(self._tf, self._base_frame_name,self._frame_name, time)
        box_center_offset = config['box_center_offset']
        box_pos = np.asarray(np.dot(pose,np.array([box_center_offset[0],box_center_offset[1],box_center_offset[2],1]))).ravel()[:3]
        _, box_q = transform_to_pq(pose)
        pose = pq_to_transform(self._tf,box_pos,box_q)


        return pose, box_pos, box_q

    # Computes a list of "pushes", a push contains a pre-push pose, 
    # a push action (goal position a push starting from a pre-push pose) 
    # and its respective name
    # It also returns the current box pose, and special reset_push
    def get_pushes(self):

        success = False
        max_trials = 200
        trial_count = 0

        pos = pose = time = ee_pose = box_q = box_pos =  None

        while trial_count < max_trials and not success:

            try:
                pose, box_pos, box_q = self.get_pose()

                time = self._tf.getLatestCommonTime(self._base_frame_name, 'left_gripper')
                ee_pose = get_pose(self._tf, self._base_frame_name,'left_gripper', time)

                ee_pos, q_ee = transform_to_pq(ee_pose)

                reset_pos, reset_q = self.get_reset_pose()

                success = True
            except Exception as e:
                print "Failed to get required transforms", e
                trial_count += 1


        
        if success:
            pre_push_offset = config['pre_push_offsets']

            length_div2 = config['box_type']['length']/2.0
            breadth_div2 = config['box_type']['breadth']/2.0
            
            x_box = random.uniform(-length_div2,length_div2)
            z_box = random.uniform(-breadth_div2,breadth_div2)

            pre_positions = np.array([[pre_push_offset[0]    , pre_push_offset[1],  z_box,               1], # right-side of the object
                                  [-pre_push_offset[0]   , pre_push_offset[1],  z_box,               1], # lef-side of the object
                                  [x_box                 , pre_push_offset[1],  pre_push_offset[2],  1], # front of the object
                                  [x_box                 , pre_push_offset[1], -pre_push_offset[2],  1]])  # back of the object

            push_locations = np.array([[0    , pre_push_offset[1],  z_box,               1], # right-side of the object
                                       [0   , pre_push_offset[1],  z_box,               1], # lef-side of the object
                                       [x_box                 , pre_push_offset[1],  0,  1], # front of the object
                                       [x_box                 , pre_push_offset[1],  0,  1]])  # back of the object


            pushes = []
            count = 0
            for pos_idx in range(len(pre_positions)):

                pre_position = pre_positions[pos_idx] # w.r.t to box
                push_position = push_locations[pos_idx] # w.r.t to box

                # "position" is relative to the box        
                pre_push_pos1 = np.asarray(np.dot(pose,pre_position)).ravel()[:3]
                pre_push_dir0 = pre_push_pos1 - ee_pos
                pre_push_dir0[2] = 0
                pre_push_pos0 = ee_pos + pre_push_dir0

                # Pushing towards the center of the box
                push_action = np.asarray(np.dot(pose,push_position)).ravel()[:3] # w.r.t to base frame now

                push_xz = np.array(push_position[0],push_position[2])
                pushes.append({'poses': [{'pos': pre_push_pos0, 'ori': box_q}, {'pos': pre_push_pos1, 'ori': box_q}], 'push_action': push_action, 'push_xz': push_xz, 'name' : 'pre_push%d'%(count,)})

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



class PushMachine(object):

    def __init__(self, robot_interface, sample_start_index=None):

        self._push_counter = 0
        self._box = BoxObject()
        self._robot = robot_interface

        self._states = {'RESET': 0, 'PUSH' : 1}
        self._state = self._states['RESET']

        self._record_sample = RecordSample(robot_interface=robot_interface, 
                                           task_interface=BoxObject(),
                                           data_folder_path=None,
                                           data_name_prefix='push_data',
                                           num_samples_per_file=5)


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
            #success = self.reset_box(reset_push)
            os.system("spd-say 'Please reset the box, Much appreciated dear human'")
            raw_input("Press enter to continue...")

        elif self._state == self._states['PUSH']:
            print "Moving to pre-push position ..."
                    
            # There might be a sequence of positions prior to a push action
            goals = self.pack_push_goals(pushes[idx])

            success = self.goto_goals(goals=goals, record=True, push = pushes[idx])

            goals.reverse()
            success = self.goto_goals(goals[1:])

            if success:
                self._push_counter += 1
            
            idx = (idx+1)%(len(pushes))
        else:
            print "UNKNOWN STATE"


        return idx, success

    def pack_push_goals(self,push):

        goals = []
        for goal in push['poses']:
            goals.append(goal)

        push_action = push['push_action']

        goals.append({'pos': push_action, 'ori': None})


        return goals

    def goto_goals(self,goals, record=False, push = None):

        c = 0
        success = False
        for i in range(len(goals)-1):

            goal = goals[i]

            success = self.goto_pose(goal_pos=goal['pos'], goal_ori=None)

            if not success:
                return False

        # Push is always the last
        goal = goals[len(goals)-1]
        if record and push:
            print "Gonna push the box ..."
            # self._record_sample.start_record(push)
            
            self._record_sample.record_once(task_action=push)
            success = self.goto_pose(goal_pos=goal['pos'], goal_ori=None)
            self._record_sample.record_once(task_action=None, task_status=success)


        return success



    def run(self):

        push_finished = True
        pre_push_finished = True

        t = 0

        rate = rospy.Rate(10)

        idx = 0

        while not rospy.is_shutdown():# and not finished:

            pushes = None
            box_pose = None

            print "Moving to neutral position ..."
                        
            self._robot.untuck_arm()

            
            pushes, box_pose, reset_push = self._box.get_pushes()
            self._box._last_pushes = pushes

            if pushes:

                self.compute_next_state(idx)

                idx, success = self.goto_next_state(idx, pushes, box_pose, reset_push)

            rate.sleep()

        #this if for saving files in case keyboard interrupt happens
        self._record_sample.save_data_now()

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

            success = success and self.goto_pose(goal_pos=goal['pos'], goal_ori=None)


        ee_pos, _ = self._robot.get_ee_pose()

        success = success and self.goto_pose(goal_pos=ee_pos+reset_push['push_action'], goal_ori=None)

        return success


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Data collection for push manipulation')
    
    parser.add_argument('-n', '--sample_start_index', type=int, help='start index of sample collection')
    
    args = parser.parse_args()

    rospy.init_node('poke_box', anonymous=True)
    from aml_robot.baxter_robot import BaxterArm
    limb = 'left'
    arm = BaxterArm(limb)
    
    push_machine = PushMachine(robot_interface=arm, sample_start_index=args.sample_start_index)

    print "calling run"

    push_machine.run()
   
    

    

