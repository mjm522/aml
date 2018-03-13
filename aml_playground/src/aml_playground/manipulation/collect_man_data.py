#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pybullet as pb
from utils import rotz, roty, rotx
from aml_io.io_tools import save_data,load_data
from omni_interface.phantom_omni import PhantomOmni
from aml_rl_envs.utils.collect_demo import draw_trajectory
from aml_rl_envs.hand.hand_obst_env import HandObstacleEnv
from aml_playground.manipulation.config import HAND_OBJ_CONFIG
from aml_rl_envs.utils.data_utils import save_csv_data, load_csv_data


class CollectManData():

    def __init__(self, env=None, use_ph_omni=False):

        HAND_OBJ_CONFIG['ctrl_type'] = 'torq'

        if env is None:
            # self._env = HandObjEnv(action_dim=18, randomize_box_ori=False, keep_obj_fixed=True, config=HAND_OBJ_CONFIG)
            self._env = HandObstacleEnv(action_dim=18, randomize_box_ori=False, keep_obj_fixed=False, config=HAND_OBJ_CONFIG)
        else:
            self._env = env

        self._object = self._env._object

        self._angle_div = 30.

        self._data_root = os.environ['AML_DATA'] + '/aml_playground/manipulation/'

        self._sp_obj_file_name = self._data_root + 'surface_point_obj.csv'
        
        self._sn_obj_file_name = self._data_root + 'surface_normal_obj.csv'

        self._data_file_name = self._data_root + 'demo1/collect_man_data.pkl'

        self._data = []

        self._use_ph_omni= use_ph_omni

        self._calib_pos = None 

        self._calib_ori = None

        self._calibrated = False

        data = self._env._hand.get_ee_states()

        self._old_ee_pos = data[0][1]

        if self._use_ph_omni:

            import rospy

            rospy.init_node('teleop_bullet', anonymous=True)

            self._ph_omni = PhantomOmni()

            time.sleep(1)

        print "Done"


    def discretize_obj_surface(self):
        '''
        keep the object away from all other things in the env
        run this function to get a discretized representation of
        the object in its own frame
        '''

        obj_pos, obj_ori = self._object.get_curr_state(ori_type='quat')[0:2]

        rays_from_centre = []

        theta = np.arange(0., 2.*np.pi, self._angle_div*np.pi/180.)

        start_ray = np.array([1., 0., 0])

        sp_obj_frame = []
        
        sn_obj_frame = []

        for k in theta:
            
            for j in theta:
                
                for i in theta:

                    ray = np.dot( np.dot( np.dot(rotz(k), roty(j)), rotx(i) ), start_ray)

                    ray = np.dot(obj_ori, ray) + obj_pos

                    # draw_trajectory(obj_pos, ray)

                    # print "objectUniqueId \t linkIndex \t hit fraction \t hit position \t hit normal"

                    ray_test_data = pb.rayTest(ray, obj_pos)[0]

                    sp_obj, _ = self._env.transfer_point_from_world_to_obj(ray_test_data[3])
                    
                    sp_obj_frame.append(sp_obj)

                    sn_obj, _ = self._env.transfer_point_from_world_to_obj(ray_test_data[4])
                    
                    sn_obj_frame.append(sn_obj)


        save_csv_data(self._sp_obj_file_name, sp_obj_frame)
        
        save_csv_data(self._sn_obj_file_name, sn_obj_frame)

    def load_surface_points_object(self):

        sp_obj_frame  = load_csv_data(self._sp_obj_file_name)
        
        sn_obj_frame = load_csv_data(self._sn_obj_file_name)

        return sp_obj_frame, sn_obj_frame


    def collect_data(self):

        data_point = {}

        data_point['obj_kin_state']   = self._env.get_obj_kin_state(ori_type='quat')
        
        data_point['obj_dyn_state']   = self._env.get_obj_dyn_state()
        
        data_point['robot_state']     = self._env.get_robot_curr_state()
        
        data_point['cp_obj_table']    = self._env.get_contact_points_object_table()
        
        data_point['cp_obj_obstacle'] = self._env.get_contact_points_object_obstacle()
        
        data_point['cp_obj_robot']    = self._env.get_contact_points_robot_object()

        self._data.append(data_point)


    def get_reaction_force(self, finger_idx=0):

        data = self._env.get_contact_points_robot_object()

        if not data[finger_idx]['cn_force']:

            return np.zeros(3)

        force = np.asarray(data[finger_idx]['cn_on_block'][0]) * data[finger_idx]['cn_force'][0]

        return -force


    def transform_demo_device_to_bullet(self):

        return self._ph_omni.get_ee_pose_calib_space()


    def calibration(self, finger_idx=0):

        robot_curr_state = self._env.get_robot_curr_state()

        ee_pos = robot_curr_state['pos_ee'][finger_idx]

        ee_ori = robot_curr_state['ori_ee'][finger_idx] 

        self._ph_omni.calibration(ee_pos, ee_ori)

        self._calibrated = self._ph_omni._calibrated


    def move_relative_to_demo_device(self, finger_idx=0):

        if not self._calibrated:
            
            return

        robot_curr_state = self._env.get_robot_curr_state()

        pos, ori = self.transform_demo_device_to_bullet()

        # draw_trajectory(self._old_ee_pos, pos)

        self._old_ee_pos = pos

        cmd = self._env.get_ik(finger_idx, pos, ori)

        # self._env._hand.applyAction(finger_idx, cmd)

        self._env._hand.set_fin_joint_state(finger_idx, cmd)

        # self._env._object.set_obj_pose(new_ee_pos, new_ori)


    def run(self):

        start_collection = False

        while True:

            keyboard_events = pb.getKeyboardEvents()

            if self._use_ph_omni:

                self.move_relative_to_demo_device()

                # self.transform_demo_device_to_bullet()

                if self._ph_omni._omni_bt_state['white_bt']:

                    print "Calibrating ..."

                    self.calibration()

            # obj = self._env.get_obj_curr_state(ori_type='quat')
            # values = self._env.get_contact_points_robot_object()[1]['cp_on_block']

            # # for val in values:
            # #     tmp = self._env.transfer_point_from_world_to_obj(val)[0]
            # #     print "World 1 \t", val
            # #     print "Obj 1 \t", tmp

            #press s key to start data collection
            if 115L in keyboard_events.keys():

                start_collection = True

            #press e key to start data collection
            elif 101L in keyboard_events.keys():

                start_collection = False

                self._calibrated = False

                if self._data:

                    save_data(data=self._data, filename=self._data_file_name)

                self._data = []

                self._env._reset(obj_base_fixed=False)

               
            
            if start_collection:
                
                self.collect_data()

            self._env.simple_step()

            if self._use_ph_omni:

                pass

                # self._ph_omni.omni_force_feedback(self.get_reaction_force())

if __name__ == '__main__':
    
    cmd = CollectManData(use_ph_omni=True)
    # cmd.discretize_obj_surface()
    cmd.run()