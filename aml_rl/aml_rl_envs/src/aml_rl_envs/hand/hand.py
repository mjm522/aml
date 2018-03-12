import os
import copy
import math
import itertools
import numpy as np
import pybullet as pb
from os.path import exists, join
from aml_rl_envs.aml_rl_robot import AMLRlRobot


class Hand(AMLRlRobot):

    def __init__(self, config, pos = [1.75, 0., 2.75], ori = [0., 0., 0., 1], j_pos=[np.pi/3, 0.0, 0.0, 0.000], hand_choice = 'four_finger'):

        AMLRlRobot.__init__(self, config)

        if hand_choice == 'pincer':
            
            self._robot_id = pb.loadURDF(join(self._urdf_root_path, "hand/pincer_arm.urdf"), useFixedBase=True, globalScaling=1)

            self._num_fingers = 2    
        
        else:
            
            self._robot_id = pb.loadURDF(join(self._urdf_root_path, "hand/four_finger_hand.urdf"), useFixedBase=True, globalScaling=1)

            self._num_fingers = 4

        
        self._finger_radius = 0.05 # from urdf
        
        self._num_joints_per_finger = 4
        
        self._ee_indexs = [self._num_joints_per_finger*i+(self._num_joints_per_finger-1) for i in range(self._num_fingers)]

        if len(j_pos) < pb.getNumJoints(self._robot_id):
            
            self._jnt_postns = j_pos[:self._num_joints_per_finger]*self._num_fingers
        
        elif len(j_pos) > pb.getNumJoints(self._robot_id):
            
            self._jnt_postns = j_pos[:self._num_joints_per_finger]
        
        else:
            
            self._jnt_postns = j_pos

        self.reset(pos, ori)

    def reset(self, pos = [1.5, 0., 2.75], ori = [0., 0., 0., 0]):

        self.set_ctrl_mode()
        
        pb.resetBasePositionAndOrientation(self._robot_id, pos, ori)
        
        self._ee_pos, self._ee_ori, _, _  = self.get_ee_states()
        
        self._motor_names   = [[] for _ in range(self._num_fingers)]
        
        self._motor_indices = [[] for _ in range(self._num_fingers)]

        self._num_joints = len(self._jnt_indexs)

        assert self._num_joints == self._num_fingers*(self._num_joints_per_finger-1)

        tmp_indices = np.asarray(self._jnt_indexs).reshape(self._num_fingers, self._num_joints_per_finger-1).tolist()

        for k in range(self._num_fingers):

            for jnt_idx in tmp_indices[k]:
                
                jnt_info = pb.getJointInfo(self._robot_id, jnt_idx)
                
                q_index = jnt_info[2]

                self._motor_names[k].append(str(jnt_info[1]))
                
                self._motor_indices[k].append(jnt_idx)

        pb.setRealTimeSimulation(0)

        self.enable_force_torque_sensors()
        
        self.get_observation()

    def set_joint_state(self, finger_idx, jnt_pos):

        for jp in range (len(jnt_pos)):

            joint_idx = self._motor_indices[finger_idx][jp]

            pb.resetJointState(self._robot_id, joint_idx, jnt_pos[jp])


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

    def apply_action(self, finger_idx, motor_commands, Kp=None):

        for action in range (len(motor_commands)):

            motor = self._motor_indices[finger_idx][action]

            self.apply_ctrl(motor, action, Kp)


    def inv_kin(self, finger_idx, ee_pos, ee_ori=None):

        return self.get_ik(self._ee_indexs[finger_idx], 
                            ee_pos, ee_ori)[3*finger_idx:3*finger_idx+3]


    def get_finger_limits(self):

        lower_limit = [np.zeros(4) for _ in range(self._num_fingers)]
        
        upper_limit = [np.zeros(4) for _ in range(self._num_fingers)]

        for j in range(self._num_fingers):

            for k in range(self._num_joints_per_finger):
                
                lower_limit[j][k] = pb.getJointInfo(self._robot_id, 4*j+k)[8]
                
                upper_limit[j][k] = pb.getJointInfo(self._robot_id, 4*j+k)[9]
                
        return {'lower': lower_limit, 'upper':upper_limit}


    def get_ee_states(self, as_tuple=False):

        ee_poss = []
        ee_oris = []
        ee_vels = []
        ee_omgs = []

        for k in range(self._num_fingers):

            link_state = pb.getLinkState(self._robot_id, self._ee_indexs[k], computeLinkVelocity = 1)

            if as_tuple:
                
                ee_poss.append(link_state[0])
                ee_oris.append(link_state[1])
                ee_vels.append(link_state[6])
                ee_omgs.append(link_state[7])
            
            else:
                
                ee_poss.append(np.asarray(link_state[0]))
                ee_oris.append(np.asarray(link_state[1]))
                ee_vels.append(np.asarray(link_state[6]))
                ee_omgs.append(np.asarray(link_state[7]))

        return ee_poss, ee_oris, ee_vels, ee_omgs


    def get_jnt_states(self):

        jnt_poss = [ [] for _ in range (self._num_fingers)]
        jnt_vels = [ [] for _ in range (self._num_fingers)]
        jnt_reaction_forces =  [ [] for _ in range (self._num_fingers)]
        jnt_applied_torques  = [ [] for _ in range (self._num_fingers)]

        for finger_idx in range(self._num_fingers):

            for jnt_idx in self._motor_indices[finger_idx]:

                jnt_state = pb.getJointState(self._robot_id, jnt_idx)
                jnt_poss[finger_idx].append(jnt_state[0])
                jnt_vels[finger_idx].append(jnt_state[1])
                jnt_reaction_forces[finger_idx].append(jnt_state[2])
                jnt_applied_torques[finger_idx].append(jnt_state[3])


        return jnt_poss, jnt_vels, jnt_reaction_forces, jnt_applied_torques


    def get_mass_matrix(self, finger_idx, jnt_poss):

        jnt_pos = list(itertools.chain.from_iterable(jnt_poss))

        Mq = np.asarray(pb.calculateMassMatrix(bodyUniqueId=self._robot_id, objPositions=jnt_pos))

        #assuming equal finger joints
        active_joints = Mq.shape[1]/self._num_fingers
        
        return Mq[active_joints*finger_idx:active_joints*(finger_idx+1), 
                  active_joints*finger_idx:active_joints*(finger_idx+1)]


    def get_jacobians(self, jnt_poss):

        if len(jnt_poss[0]) != self._num_joints_per_finger:
            
            jnt_pos = np.asarray(jnt_poss).reshape(self._num_fingers, self._num_joints_per_finger-1)

            jnt_pos = np.hstack([jnt_poss, np.zeros(self._num_fingers)[:,None]])

            jnt_pos = jnt_pos.reshape(self._num_fingers*self._num_joints_per_finger).tolist()
        
        else:
            
            jnt_pos = np.asarray(jnt_poss).reshape(self._num_fingers*self._num_joints_per_finger).tolist()


        lin_jacs = []
        
        ang_jacs = []

        for k in range(self._num_fingers):

            linear_jacobian, angular_jacobian = pb.calculateJacobian(bodyUniqueId=self._robot_id,
                                                                linkIndex=self._ee_indexs[k],
                                                                localPosition=[0.,0.,0.],
                                                                objPositions=jnt_pos,
                                                                objVelocities=np.zeros(len(jnt_pos)).tolist(),
                                                                objAccelerations=np.zeros(len(jnt_pos)).tolist()
                                                                )
            lin_jacs.append(np.asarray(linear_jacobian))

        return lin_jacs


    def get_finger_jacobian(self, finger_idx, jnt_poss, local_point=None):

        if local_point is None:
            local_point = [0.,0.,0.]

        jnt_pos = list(itertools.chain.from_iterable(jnt_poss))

        jnt_pos = jnt_pos[:3] + [0.] +  jnt_pos[3:] + [0.]

        linear_jacobian, angular_jacobian = pb.calculateJacobian(bodyUniqueId=self._robot_id,
                                                                linkIndex=self._ee_indexs[finger_idx],
                                                                localPosition=local_point,
                                                                objPositions=jnt_pos,
                                                                objVelocities=np.zeros(self._num_joints+2).tolist(),
                                                                objAccelerations=np.zeros(self._num_joints+2).tolist()
                                                                )

        Jee = np.asarray(linear_jacobian)
        
        cols = Jee.shape[1]/self._num_fingers

        return Jee[:, cols*finger_idx:cols*(finger_idx+1)]


    def get_hand_jacobian(self):

        jnt_poss = self.get_jnt_states()[0]

        jnt_pos = list(itertools.chain.from_iterable(jnt_poss))

        hand_jacobian = None

        for k in range(self._num_fingers):

            linear_jacobian, angular_jacobian = pb.calculateJacobian(bodyUniqueId=self._robot_id,
                                                                linkIndex=self._ee_indexs[k],
                                                                localPosition=[0.,0.,0.],
                                                                objPositions=jnt_pos,
                                                                objVelocities=np.zeros(self._num_joints).tolist(),
                                                                objAccelerations=np.zeros(self._num_joints).tolist()
                                                                )
            if hand_jacobian is None:
                
                hand_jacobian = np.asarray(linear_jacobian)
            
            else:
                
                hand_jacobian = np.vstack([hand_jacobian, np.asarray(linear_jacobian)])

        return hand_jacobian
      

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