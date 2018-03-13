import os
import copy
import math
import itertools
import numpy as np
import pybullet as pb
from os.path import exists, join
from aml_rl_envs.aml_rl_hand import AMLRlHand

class Hand(AMLRlHand):

    def __init__(self, config, pos = [1.75, 0., 2.75], ori = [0., 0., 0., 1], j_pos=[np.pi/3, 0.0, 0.0, 0.000], hand_choice = 'four_finger'):

        if hand_choice == 'pincer':
            
            self._robot_id = pb.loadURDF(join(config['urdf_root_path'], "hand/pincer_arm.urdf"), useFixedBase=True, globalScaling=1)

            self._num_fingers = 2    
        
        else:
            
            self._robot_id = pb.loadURDF(join(config['urdf_root_path'], "hand/four_finger_hand.urdf"), useFixedBase=True, globalScaling=1)

            self._num_fingers = 4

        
        self._finger_radius = 0.05 # from urdf

        if self._num_fingers == 2:

            self._finger_jnt_indices = [[0,1,2], [4,5,6]]
        
        else:

            self._finger_jnt_indices = [[0,1,2], [4,5,6], [8,9,10], [12,13,14]]

        AMLRlHand.__init__(self, config, self._num_fingers, self._finger_jnt_indices, self._robot_id)
       
        self._ee_indexs = [3,7,11,15]

        for k in range(self._num_fingers):

            self.set_fin_joint_state(k, j_pos)

        self.reset(pos, ori)

    def reset(self, pos = [1.5, 0., 2.75], ori = [0., 0., 0., 0]):

        self.set_base_pose(pos, ori)

        self.setup_hand()
        
        self._ee_pos, self._ee_ori, _, _  = self.get_ee_states()

        self.enable_force_torque_sensors()
        
        self.get_observation()


    def get_action_dim(self):
        
        return 6 #position x,y,z and roll/pitch/yaw euler angles of end effector

    def get_obs_dim(self):
        
        return len(self.get_observation())

    def get_observation(self):
        
        observation = []

        ee_poss, ee_oris, ee_vels, ee_omgs  = self.get_ee_states()

        jnt_poss, jnt_vels, jnt_reaction_forces, jnt_applied_torques = self.get_jnt_states()

        self._Jee_old = self.get_jacobians(jnt_poss=jnt_poss)
        
        for k in range(self._num_fingers):
        
            observation.extend(jnt_poss[k])
            
            observation.extend(jnt_vels[k])

        return observation

    def compute_impedance_ctrl(self, finger_idx, Kp, goal_pos, goal_vel=np.zeros(3), dt=0.01):

        jnt_poss, jnt_vels, jnt_reaction_forces, jnt_applied_torques = self.get_jnt_states()

        Jee = self.get_finger_jacobian(finger_idx, jnt_poss)
        
        Mq  = self.get_mass_matrix(finger_idx=finger_idx, jnt_poss=jnt_poss)

        ee_force = np.asarray(jnt_reaction_forces[finger_idx][-1][:3])

        curr_pos, curr_ori, curr_vel, curr_omg  = self.get_ee_states()

        delta_pos      = goal_pos - curr_pos[finger_idx]
        
        delta_vel      = goal_vel - curr_vel[finger_idx]

        # delta_ori      = quatdiff(goal_ori, curr_ori)
        # delta_omg      = np.zeros(3)

        #Compute cartesian space inertia matrix
        Mq_inv    = np.linalg.inv(Mq)
        
        Mcart_inv = np.dot(np.dot(Jee, Mq_inv), Jee.transpose())
        
        Mcart     = np.linalg.pinv(Mcart_inv, rcond=1e-3)

        #inertia shaping, as same as eee inertia
        Md_inv  = Mcart_inv #(np.linalg.inv(self._Md))

        #from morteza slide
        xdd = np.zeros(3)
        
        tmp = np.dot(Mcart, Md_inv)

        f = ee_force + np.dot(Mcart, xdd) + np.dot(tmp, (np.multiply(Kp, delta_pos) + np.multiply(np.sqrt(Kp), delta_vel)) ) - np.dot(tmp, ee_force)

        tau_task = np.dot( np.dot(Jee.transpose(), Mcart),  f)

        return tau_task