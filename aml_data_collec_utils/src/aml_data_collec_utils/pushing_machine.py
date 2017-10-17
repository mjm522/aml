#!/usr/bin/env python

# A simply python controller for Baxter compatible with GPS (TODO)
import tf
import os
import rospy
import argparse
import numpy as np
import quaternion
import aml_calib
import camera_calib
from aml_robot.baxter_robot import BaxterArm
from aml_data_collec_utils.box_object import BoxObject
from aml_data_collec_utils.core.sample import Sample
from aml_data_collec_utils.core.data_recorder import DataRecorder
from aml_data_collec_utils.config import config
from aml_data_collec_utils.baxter_reset_box import fsm_reset


class PushMachine(object):

    def __init__(self, robot_interface, sample_start_index=None):

        self._push_counter = 0
        self._box = BoxObject()
        self._robot = robot_interface
        self._right_arm = BaxterArm('right')

        self._states = {'RESET': 0, 'PUSH' : 1}
        self._state = self._states['RESET']

    def send_left_arm_away(self):
        self._robot.move_to_joint_position(np.array([ 0.21935925, -0.80380593,  0.06902914,  0.837937  ,  0.00421845, 1.34721863,  0.4314321 ]))

    def compute_next_state(self,idx):

        # Decide next state
        if idx == 0 and self._push_counter > 0 and self._state != self._states['RESET']:
            self._state = self._states['RESET']
        else:
            self._state = self._states['PUSH']


    def reset(self):
        print "RESETING WITH NEW POSE"
        #success = self.reset_box(reset_push)

        self.send_left_arm_away()

        order_of_sweep = ['left', 'back', 'front', 'right']
        fsm_reset(self._right_arm, order_of_sweep, rate=15)

        self._robot.untuck_arm()
        os.system("spd-say 'Reseting box without human supervision'")
        #os.system("spd-say 'Please reset the box, Much appreciated dear human'")
        #raw_input("Press enter to continue...")

    def apply_push(self,push):

        print "Moving to pre-push position ..."
                    
        # There might be a sequence of positions prior to a push action
        goals = self.pack_push_goals(push)

        start = rospy.Time.now()

        success = self.goto_goals(goals=goals, record=True, push = push)

        goals.reverse()
        success = success and self.goto_goals(goals[1:]) 

        self._robot.untuck_arm()

        timeelapsed = rospy.Time.now() - start

        print "TIME ELAPSED: ", timeelapsed.to_sec()
        
        if success:
            self._push_counter += 1

        return success


    def apply_push2(self,push_u):

        pushes, box_pose, reset_push = self._box.get_push(push_u)
        self._box._last_pushes = pushes

        success = False
        if pushes:
            print pushes[0]
            sucess = self.apply_push(pushes[0])


        return success


    def apply_push3(self, x_box, z_box):

        pushes, box_pose, reset_push = self._box.get_push(x_box, z_box)
        self._box._last_pushes = pushes

        success = False
        if pushes:
            print pushes[0]
            sucess = self.apply_push(pushes[0])


        return success


    # def sample_push(self):

    #     bw = config['box_type']['length']*config['scale_adjust']
    #     bh = config['box_type']['breadth']*config['scale_adjust']

    #     u = np.random.rand()

    #     edge_point, side, edge_vec = get_box_edge2(u,bw,bh)  # w.r.t box frame

    #     x_box, z_box = [edge_point[0],edge_point[1]]


    #     pre_position = np.array([x_box, pre_push_offset[1], z_box])

    #     box_center = np.array([0, pre_push_offset[1], 0])

    #     centre_vec = box_center - pre_position

    #     edge_vec /= np.linalg.norm(edge_vec)
    #     centre_vec /= np.linalg.norm(centre_vec)

    #     push_vec = centre_vec - np.dot(centre_vec,edge_vec)*edge_vec

    #     theta_offset = 2*np.random.rand()*(np.pi/6.0) - (np.pi/6.0)

    #     theta = np.fmod((np.arctan2(push_vec[1],push_vec[0])) + theta_offset,2*np.pi)

        



    def goto_next_state(self,idx,push, box_pose, reset_push):

        success = True
        pushed = False

        # Take machine to next state
        if self._state == self._states['RESET']:
            self.reset()

        elif self._state == self._states['PUSH']:
            
            self.apply_push(push)
            idx = (idx+1)%100
        else:
            print "UNKNOWN STATE"


        return idx, success

    def pack_push_goals(self, push):

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
        
            success = self.goto_pose(goal_pos=goal['pos'], goal_ori=None)
            


        return success

    def on_shutdown(self):
        pass
        # send_pos_cmd_pisa_hand()
        #this if for saving files in case keyboard interrupt happens
        # self._record_sample.save_data_now()

    def run(self):

        push_finished = True
        pre_push_finished = True

        t = 0

        rate = rospy.Rate(10)

        idx = 0

        self._robot.untuck_arm()

        rospy.on_shutdown(self.on_shutdown)


        while not rospy.is_shutdown():# and not finished:

            push = None
            box_pose = None

            print "Moving to neutral position ..."
                        
            
            self._robot.set_joint_position_speed(5.0)
            
            print "Xi:", self._box.get_effect()

            pushes, box_pose, reset_push = self._box.get_push(np.random.rand())

            self._box._last_pushes = pushes

            if pushes:

                print "Action:", pushes[0]

                self.apply_push(pushes[0])

            print "Xf:", self._box.get_effect()

            rate.sleep()

        

    def goto_pose(self,goal_pos, goal_ori, ntrials = 1, disturb_on_fail = True): 

        start_pos, start_ori = self._robot.get_ee_pose()

        if goal_ori is None:
             goal_ori = start_ori

        sucess = False

        goal_pos_test = goal_pos
        goal_ori = quaternion.as_float_array(goal_ori)
        print "Type goal ", type(goal_pos_test), goal_pos_test.shape
        for i in range(ntrials):

            # goal_pos_test = (goal_pos + np.random.randn(3)*0.005).astype(float)
            success, js_pos = self._robot.ik(goal_pos_test,goal_ori)

            print "Trying to push... Status:", success

            if success:
                self._robot.move_to_joint_position(js_pos)

                break
            else:
                noise = np.random.randn(3)*0.025
                noise[2] = 0.0

                goal_pos_test = goal_pos + noise
                print "Type goal ", type(goal_pos_test), goal_pos_test.shape
                print "Couldnt find a solution, trial %d disturb_flag: ", i, disturb_on_fail

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
    limb = 'left'
    arm = BaxterArm(limb)
    
    push_machine = PushMachine(robot_interface=arm, sample_start_index=args.sample_start_index)

    print "calling run"

    push_machine.run()
   
    

    

