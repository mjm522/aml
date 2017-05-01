#!/usr/bin/env python

import rospy
import numpy as np
import quaternion
from functools import partial
from aml_robot.baxter_robot import BaxterArm
from aml_io.convert_tools import string2image
from aml_data_collec_utils.config import config
from aml_visual_tools.visual_tools import show_image
from aml_data_collec_utils.box_object import BoxObject
from aml_services.srv import PredictAction, PredictState
from aml_dl.mdn.utilities.get_data_from_files import get_data_from_files
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
    rospy.wait_for_service('predict_action')
    try:
        predict_action_service = rospy.ServiceProxy('predict_action', PredictAction)
        response = predict_action_service(curr_state, tgt_state)
        return response.action[0], response.action[1], response.sigma
    except rospy.ServiceException, e:
        print "Service call to predict_action failed: %s"%e
    # import random
    # length_div2 = config['box_type']['length']*config['scale_adjust']/2.0
    # breadth_div2 = config['box_type']['breadth']*config['scale_adjust']/2.0
    
    # x_box = length_div2#random.uniform(0.,length_div2) # w.r.t box frame
    # z_box = 0.#random.uniform(0.,breadth_div2) # w.r.t box frame
    # x_box = response[0]
    # z_box = response[1]

    # return x_box, z_box

def get_data(file_inidices, model_type='siam', string_img_convert=True):

    tmp_x, data_y = get_data_from_files(data_file_range=file_inidices, model_type=model_type)

    data_x = []
    if string_img_convert:
        for x_image in tmp_x:
            #x_image[1][0] is the image at t_1 and x_image[0][0] is the iamge at t
            data_x.append(np.transpose(string2image(x_image[1][0]), axes=[2,1,0]).flatten())
            # data_x.append((np.transpose(string2image(x_image[0][0]), axes=[2,1,0]).flatten(), 
            #                np.transpose(string2image(x_image[1][0]), axes=[2,1,0]).flatten()))
    else:
        data_x = tmp_x

    return np.asarray(data_x), np.asarray(data_y)


class StochasticPushMachine(BoxObject):

    def __init__(self, robot_interface, pushing_limb_gripper='left_gripper', offset=5.):
        
        self._robot = robot_interface
        self._push_gripper_name = pushing_limb_gripper
        BoxObject.__init__(self)
        self._offset_scale = offset #offset for prepush location
        self._list_br_poses = []
        update_period = rospy.Duration(1.0/100.)
        rospy.Timer(update_period, partial(self.visualize))


    def send_left_arm_away(self):
        self._robot.move_to_joint_position(np.array([ 0.21935925, -0.80380593,  0.06902914,  0.837937  ,  0.00421845, 1.34721863,  0.4314321 ]))


    def visualize(self, event):
        if not self._list_br_poses:
            return
        else:
            for br_frame in self._list_br_poses:
                now = rospy.Time.now()
                self._br.sendTransform(br_frame['pos'], br_frame['ori'], now, br_frame['frame_name'], br_frame['base_name'])



    def add_to_br_poses(self, pos, ori, frame_name, base_name='base'):
        br_frame = {'pos':pos, 'ori':ori, 'frame_name':frame_name, 'base_name': base_name}
        self._list_br_poses.append(br_frame)

    def get_side(self,x,z):

        abs_x = np.abs(x)
        abs_z =  np.abs(z)

        
        RIGHT = 0
        LEFT = 1
        FRONT = 2
        BACK = 3
        NONE = 4

        side = NONE

        # either left or right
        if abs_x > abs_z:
            if x > 0:
                side = RIGHT
            else:
                side = LEFT
        else:
            if z > 0:
                side = FRONT
            else:
                side = BACK

        return side


    def compute_push_location(self, tgt_box_pose, image_input=False, time_out=5.):

        print "CAME HERE ***************************************************** 1"
        pre_push_offset = config['pre_push_offsets']

        print "CAME HERE ***************************************************** 2"
        box_pose, box_pos, box_ori = self.get_pose()
        
        print "CAME HERE ***************************************************** 3"
        if image_input:
            print "CAME HERE ***************************************************** 4"
            curr_state =  self.get_curr_image() #tgt_box_pose[0]
            print "CAME HERE ***************************************************** 5"
            # show_image(curr_state)
            # print type(curr_state)
            # print curr_state.shape
            # print type(tgt_box_pose)
            # print tgt_box_pose.shape
            # raw_input()
        else:
            curr_state = np.hstack([box_pos, box_ori])

        time = None
        start_time = rospy.Time.now()
        timeout = rospy.Duration(time_out) # Timeout of 'time_out' seconds
        while time is None:
            try:
                time = self._tf.getLatestCommonTime(self._base_frame_name, self._push_gripper_name)
            except Exception as e:
                if (rospy.Time.now() - start_time > timeout):
                    raise Exception("Time out reached, was not able to get *tf.getLatestCommonTime*")
                    break
                else:
                    continue
        print "CAME HERE ***************************************************** 6"

        ee_pose = get_pose(self._tf, self._base_frame_name, self._push_gripper_name, time)
        if ee_pose is None:
            print "Failed to get ee_pose returning back"
            return None, None
        
        ee_pos, q_ee = transform_to_pq(ee_pose)

        print "CAME HERE ***************************************************** 7"

        x_box, z_box, sigma = predict_action_client(curr_state=curr_state, tgt_state=tgt_box_pose)
        print "predicted push:", x_box, z_box
        print "uncertainity ", sigma

        names = ["right", "left", "front", "back", "NONE"]
        side = self.get_side(x_box,z_box)
        print "Push side: ", names[side]

        point_on_box = np.array([x_box, 0., z_box]) #with ofset on y

        offsets = np.array([[0.1, 0, 0], [-0.1, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, -0.1]])

        pre_push_pos = np.hstack([offsets[side,:] + point_on_box, 1.])
        pre_push_pos[1] = pre_push_offset[1]

        push_pos = np.hstack([np.zeros(3),1])

        # "position" is relative to the box        
        pre_push_pos1 = np.asarray(np.dot(box_pose, pre_push_pos)).ravel()[:3]
        pre_push_dir0 = pre_push_pos1 - ee_pos
        pre_push_dir0[2] = 0
        pre_push_action = ee_pos + pre_push_dir0

        push_action = np.asarray(np.dot(box_pose, push_pos)).ravel()[:3]

        self.add_to_br_poses(pos=pre_push_action, ori=q_ee, frame_name='pre-push-location')
        self.add_to_br_poses(pos=push_action, ori=q_ee, frame_name='push-location')

        k = raw_input("Sigma is %f, continue (y/n)?"%(sigma,))
        if k == 'n':
            return None, None

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
        pre_push_action, push_action = self.compute_push_location(tgt_box_pose, image_input=True)
        #this is a hack when there is no way to get the transform sending arm away and bringing
        #back tends to give the tf correctly.
        if pre_push_action is None or push_action is None:
            #self.send_left_arm_away()
            return
            
        pre_pus_stage2 = pre_push_action.copy(); pre_pus_stage2[2] = 0.01 # to go down
        push_action[2] = 0.01 #push box
        fsm_states = [pre_push_action, pre_pus_stage2, push_action, pre_pus_stage2, pre_push_action]

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

    data_file_indices = range(1,2)
    # data_X, data_Y = get_data(file_inidices=data_file_indices, model_type='fwd', string_img_convert=False)
    # data_X, data_Y = get_data(file_inidices=data_file_indices, model_type='inv', string_img_convert=False)
    data_X, data_Y = get_data(file_inidices=data_file_indices, model_type='siam', string_img_convert=True)

    while not rospy.is_shutdown():
        for data_x, data_y in zip(data_X, data_Y):
            print type(data_x)
            print len(data_x)

            print "Trying to push box to target pose \t", np.round(data_y[9:],3)
            spm.add_to_br_poses(pos=data_y[9:12], ori=np.array([data_y[13],data_y[14], data_y[15], data_y[12]]), frame_name='target_box_pose')
            status = spm.push_box(data_x)
            if status is None:
                print "Failed to go to the target pose \t", np.round(data_y[9:],3)
                print "Trying the next one."
                spm._list_br_poses = []
                continue

if __name__ == '__main__':
    main()