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
from aml_data_collec_utils.core.data_manager import DataManager
from aml_data_collec_utils.core.data_recorder import DataRecorder
from aml_data_collec_utils.config import config
from aml_data_collec_utils.baxter_reset_box import fsm_reset

from aml_services.srv import SendPisaHandCmd

def pisa_hand_service_send_pos_client(cmd):
    rospy.wait_for_service('pisa_hand_pos_cmd')
    try:
        pisa_hand_service_ = rospy.ServiceProxy('pisa_hand_pos_cmd', SendPisaHandCmd)
        pisa_hand_service_(cmd)
    except rospy.ServiceException, e:
        print "Service call to pisa_hand_pos_cmd failed: %s"%e

class PushMachine(object):

    def __init__(self, robot_interface, sample_start_index=None):

        print "Making the reset procedure, take the reset stick and make it hold it pisa hand"
        cmd = raw_input('Enter position command (between 0 and 1 to close)')
        pisa_hand_service_send_pos_client(float(cmd))

        self._push_counter = 0
        self._box = BoxObject()
        self._robot = robot_interface
        self._right_arm = BaxterArm('right')

        self._states = {'RESET': 0, 'PUSH' : 1}
        self._state = self._states['RESET']

        self._sample_recorder = DataRecorder(robot_interface=self._robot, 
                                           task_interface=self._box,
                                           data_folder_path=config['data_folder_path'],
                                           data_name_prefix='test_push_data',
                                           num_samples_per_file=5)


    def send_left_arm_away(self):
        self._robot.move_to_joint_position(np.array([ 0.21935925, -0.80380593,  0.06902914,  0.837937  ,  0.00421845, 1.34721863,  0.4314321 ]))

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

            self.send_left_arm_away()

            order_of_sweep = ['left', 'back', 'front', 'right']
            fsm_reset(self._right_arm, order_of_sweep, rate=15)

            self._robot.untuck_arm()
            os.system("spd-say 'Reseting box without human supervision'")
            #os.system("spd-say 'Please reset the box, Much appreciated dear human'")
            #raw_input("Press enter to continue...")

        elif self._state == self._states['PUSH']:
            print "Moving to pre-push position ..."
                    
            # There might be a sequence of positions prior to a push action
            goals = self.pack_push_goals(pushes[idx])

            start = rospy.Time.now()

            self._sample_recorder.start_record(task_action=pushes[idx])

            success = self.goto_goals(goals=goals, record=True, push = pushes[idx])

            goals.reverse()
            success = success and self.goto_goals(goals[1:]) 

            self._robot.untuck_arm()

            self._sample_recorder.stop_record(task_status=success)

            timeelapsed = rospy.Time.now() - start

            print "TIME ELAPSED: ", timeelapsed.to_sec()
        
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

            pushes = None
            box_pose = None

            print "Moving to neutral position ..."
                        
            

            
            pushes, box_pose, reset_push = self._box.get_pushes()
            self._box._last_pushes = pushes

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
   
    

    

