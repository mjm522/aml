#!/usr/bin/env python

import rospy
import numpy as np
import quaternion
from aml_robot.baxter_robot import BaxterArm
from aml_data_collec_utils.config import config
from aml_data_collec_utils.box_object import BoxObject
from aml_services.srv import PredictAction, PredictState
from aml_data_collec_utils.ros_transform_utils import get_pose, transform_to_pq, pq_to_transform


def predict_state_client(state, action):
    rospy.wait_for_service('predict_state')
    try:
        predict_state_service = rospy.ServiceProxy('predict_state', PredictState)
        response = predict_state_service(state, action)
        return response.next_state
    except rospy.ServiceException, e:
        print "Service call to predict_state failed: %s"%e


def predict_action_client(curr_state, tgt_state):
    # rospy.wait_for_service('predict_action')
    # try:
    #     predict_action_service = rospy.ServiceProxy('predict_action', PredictAction)
    #     response = predict_action_service(curr_state, tgt_state)
    #     return response.action
    # except rospy.ServiceException, e:
    #     print "Service call to predict_action failed: %s"%e
    import random
    length_div2 = config['box_type']['length']*config['scale_adjust']/2.0
    breadth_div2 = config['box_type']['breadth']*config['scale_adjust']/2.0
    
    x_box = length_div2#random.uniform(0.,length_div2) # w.r.t box frame
    z_box = 0.#random.uniform(0.,breadth_div2) # w.r.t box frame

    return x_box, z_box


class StochasticPushMachine(BoxObject):

    def __init__(self, robot_interface, pushing_limb_gripper='left_gripper', offset=10.):
        self._robot = robot_interface
        self._push_gripper_name = pushing_limb_gripper
        BoxObject.__init__(self)
        self._offset_scale = offset #offset for prepush location

    def send_left_arm_away(self):
        self._robot.move_to_joint_position(np.array([ 0.21935925, -0.80380593,  0.06902914,  0.837937  ,  0.00421845, 1.34721863,  0.4314321 ]))

    def compute_push_location(self, tgt_box_pose):
        pre_push_offset = config['pre_push_offsets']

        box_pose, box_pos, box_ori = self.get_pose()

        while True:
            try:
                time = self._tf.getLatestCommonTime(self._base_frame_name, self._push_gripper_name)
            except Exception as e:
                continue
            break

        ee_pose = get_pose(self._tf, self._base_frame_name, self._push_gripper_name, time)
        if ee_pose is None:
            print "Failed to get ee_pose returning back"
            return None, None
        
        ee_pos, q_ee = transform_to_pq(ee_pose)

        x_box, z_box = predict_action_client(curr_state=np.hstack([box_pos, box_ori]), tgt_state=tgt_box_pose)

        point_on_box = np.array([x_box, 0., z_box]) #with ofset on y
        pre_push_pos = np.hstack([self._offset_scale*point_on_box, 1.])
        pre_push_pos[1] = pre_push_offset[1]

        push_pos = np.hstack([np.zeros(3),1])

        # "position" is relative to the box        
        pre_push_pos1 = np.asarray(np.dot(box_pose, pre_push_pos)).ravel()[:3]
        pre_push_dir0 = pre_push_pos1 - ee_pos
        pre_push_dir0[2] = 0
        pre_push_action = ee_pos + pre_push_dir0

        push_action = np.asarray(np.dot(box_pose, push_pos)).ravel()[:3]

        now = rospy.Time.now()
        self._br.sendTransform(pre_push_action, q_ee, now, "pre-push-location", 'base')
        self._br.sendTransform(push_action, q_ee, now, "push-location", 'base')

        raw_input("Enter to continue")

        return pre_push_action, push_action

    def goto_pose(self, goal_pos, goal_ori): 

        start_pos, start_ori = self._robot.get_ee_pose()

        if goal_ori is None:
             goal_ori = start_ori

        goal_ori = quaternion.as_float_array(goal_ori)[0]
        success, js_pos = self._robot.ik(goal_pos, goal_ori)

        if success:
            self._robot.move_to_joint_position(js_pos)
        else:
            print "Couldnt find a solution"

        return success

    def push_box(self, tgt_box_pose):
        self._robot.untuck_arm()
        pre_push_action, push_action = self.compute_push_location(tgt_box_pose)
        #this is a hack when there is no way to get the transform sending arm away and bringing
        #back tends to give the tf correctly.
        if pre_push_action is None or push_action is None:
            self.send_left_arm_away()
            return
            
        pre_pus_stage2 = pre_push_action.copy(); pre_pus_stage2[2] = 0.01 # to go down
        push_action[2] = 0.01 #push box
        fsm_states = [pre_push_action, pre_pus_stage2, push_action, -pre_pus_stage2, -pre_push_action]

        for action in fsm_states:
            action_status = self.goto_pose(goal_pos=action, goal_ori=None)
            if action_status:
                continue
            else:
                break

def main():
    rospy.init_node('stochastic_push_machine', anonymous=True)
    limb = 'left'
    arm = BaxterArm(limb)
    spm = StochasticPushMachine(robot_interface=arm)
    x_tgt_list    = np.random.randn(100,7).tolist()
    while not rospy.is_shutdown():
        for tgt_pose in x_tgt_list:
            status = spm.push_box(tgt_pose)
            if status is None:
                print "Failed to go to the target pose \t", tgt_pose
                print "Trying the next one."
                continue

if __name__ == '__main__':
    main()