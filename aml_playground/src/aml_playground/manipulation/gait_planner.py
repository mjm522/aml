import time 
import copy
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import block_diag
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
from aml_rl_envs.utils.math_utils import skew
from aml_rl_envs.utils.collect_demo import plot_demo
from aml_rl_envs.hand.hand_obj_env import HandObjEnv
from aml_playground.manipulation.config import HAND_OBJ_CONFIG
from utils import compute_f_cone_approx, unit_normal, poly_area

np.random.seed(123)

class GaitPlanner():

    def __init__(self, env=None):

        if env is None:
            
            HAND_OBJ_CONFIG['ctrl_type'] = 'pos'

            self._env = HandObjEnv(config=HAND_OBJ_CONFIG, action_dim=18, randomize_box_ori=False,  keep_obj_fixed=True)
        
        else:
            
            self._env = env

        self._finger_limits = self._env.get_hand_limits()
        self._num_fingers = self._env._num_fingers
        self._num_joints_finger = self._env._num_joints_finger

        self._finger_joint_means  = np.zeros([self._num_fingers, self._num_joints_finger])
        self._finger_joint_ranges = np.zeros([self._num_fingers, self._num_joints_finger])

        for k in range(self._num_fingers):
            for j in range(self._num_joints_finger):
                self._finger_joint_means[k,j]  = 0.5*(self._finger_limits['lower'][k][j] + self._finger_limits['upper'][k][j])
                self._finger_joint_ranges[k,j] = (self._finger_limits['upper'][k][j] - self._finger_limits['lower'][k][j])


        self._opposite_fin_idx = [2,3,0,1]

        self._initial_cs = self.get_contact_points(in_obj_frame=True)


    def check_contact(self, finger_idx):

        contact_info = self._env.get_contact_points()

        if contact_info[finger_idx]['cp_on_block']:
            return True
        else:
            return False


    def get_contact_points(self, in_obj_frame=False):
        initial_cs = []
        contact_info = self._env.get_contact_points()

        for k in range(self._num_fingers):

            if contact_info[k]['cp_on_block']:
                
                contact_point = np.asarray(contact_info[k]['cp_on_block'][0])

                if in_obj_frame:

                    contact_point, _ = self._env.transfer_point_from_world_to_obj(contact_point)

                initial_cs.append( contact_point )

        return initial_cs



    def find_new_fin_pos(self, finger_to_switch):
        """
        this is a simple heuristic
        Assumption, if one finger is being switched, then
        all other three fingers are in contact.
        In case of these four fingers, there is an opposite finger for each finger
        So ideally to maintain stability, opposite finger contacts has to fall
        within their friction cones. An easy way to do this is to give the target point of
        the new finger as exactly opposite to the finger being switched.

        this heuristic takes the assumption that the object is symmetrical in x and y axis!

        Possible issue: does not check for kinematic feasibility
        """
        
        # initial_cs = self.get_contact_points()

        obj_ori = self._env.get_obj_curr_state(ori_as_euler=False)[1]

        #we need to keep the finger opposite to this
        # opp_fin_world = initial_cs[self._opposite_fin_idx[finger_to_switch]]

        # opp_fin_obj, _ = self._env.transfer_point_from_world_to_obj(opp_fin_world)

        # new_fin_pos_obj = np.hstack([-opp_fin_obj[:2], opp_fin_obj[2]])

        opp_fin_world = np.dot(obj_ori.T, self._initial_cs[finger_to_switch])

        new_fin_pos, _ = self._env.transfer_point_from_obj_to_world(opp_fin_world)

        return new_fin_pos


    def move_to_home(self, finger_idx, offset=0.2):

        home_pos  = self.find_new_fin_pos(finger_idx)

        curr_ee_pos = self._env.get_hand_ee_state()['pos'][finger_idx]

        obj_pos = self._env.get_obj_curr_state(ori_as_euler=False)[0]

        out_point_1 = (curr_ee_pos - obj_pos)

        out_point_1 = out_point_1/np.linalg.norm(out_point_1)*0.4 + curr_ee_pos

        tmp = 0.5*(curr_ee_pos + home_pos)
        
        if finger_idx == 0:
            mid_point = tmp + np.array([offset, offset, 0])
        elif finger_idx == 1:
            mid_point = tmp + np.array([-offset,offset, 0])
        elif finger_idx == 2:
            mid_point = tmp + np.array([-offset,-offset, 0])
        elif finger_idx == 3:
            mid_point = tmp + np.array([offset,-offset, 0])


        base_len = 2100

        primitive_base =  np.zeros([base_len, 3])
        
        for k in range(3):
            primitive_base[:, k] = savgol_filter(np.hstack([np.linspace(curr_ee_pos[k], out_point_1[k], base_len/3),
                                                            np.linspace(out_point_1[k], mid_point[k], base_len/3), 
                                                            np.linspace(mid_point[k], home_pos[k], base_len/3)]), base_len/2+1, 2)


        plot_demo(primitive_base, start_idx=0)


        for k in range(primitive_base.shape[0]):

            cmd = self._env._hand.inv_kin(finger_idx, primitive_base[k, :].tolist())

            self._env._hand.applyAction(finger_idx, cmd)#, Kp=np.array([1., 0.1, 0.1]))

            self._env.simple_step()

        #that means we have not yet been able to make contact
        while not self.check_contact(finger_idx):
            print "Trying to make contact!"

            cmd = np.asarray(self._env._hand.get_jnt_states()[0][finger_idx][:3]) + np.array([0, -0.001, 0.])

            self._env._hand.applyAction(finger_idx, cmd)#, Kp=np.array([1., 0.1, 0.1]))
            self._env.simple_step()

            

        
