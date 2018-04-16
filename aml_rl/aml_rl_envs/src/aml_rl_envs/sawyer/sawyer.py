import os
import sys
import copy
import math
import numpy as np
import pybullet as pb
from aml_rl_envs.aml_rl_robot import AMLRlRobot

class Sawyer(AMLRlRobot):

    def __init__(self, config, cid):

        self._config = config

        self._gripper_index = 7

        AMLRlRobot.__init__(self, config, cid)

        self.reset()

        self._joint_tags = dict(zip(self.get_joint_names(), range(pb.getNumJoints(self._robot_id, physicsClientId=self._cid))))

        self._ee_index = 16#self._joint_tags['right_poking_stick_tip_fixed']

        self._ft_sensor_jnt = self._joint_tags['right_wrist']
 
    def reset(self):

        self._robot_id = pb.loadURDF(os.path.join(self._config['urdf_root_path'],"sawyer/sawyer2_with_peg.urdf"), useFixedBase=True, physicsClientId=self._cid)

        pb.resetBasePositionAndOrientation(self._robot_id,[-0.100000,0.000000, 1.0000],[0.000000,0.000000,0.000000,1.000000], physicsClientId=self._cid)
        
        self._jnt_postns=[-5.26523437e-02, -1.18152539e+00, -2.35156250e-03,  2.05699707e+00, 3.74414063e-03,  6.67680664e-01,  3.31310840e+00]
        
        self._movable_jnts = self.get_movable_joints()
        
        self._motor_names = []
        
        self._motor_indices = []
        
        for i in self._movable_jnts:
            
            jnt_info = pb.getJointInfo(self._robot_id, i, physicsClientId=self._cid)
            
            qIndex = jnt_info[3]
            
            if qIndex > -1:

                self._motor_names.append(str(jnt_info[1]))
                
                self._motor_indices.append(i)

        self.set_ctrl_mode(jnt_postns=self._jnt_postns)


    def get_action_dim(self):

        return 9 #position x,y,z and roll/pitch/yaw euler angles of end effector

    def get_obs_dim(self):

        return len(self.get_observation())

    def get_observation(self):
        
        observation = []
        
        jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque = self.get_jnt_state()
        
        pos, ori, vel, omg  = self.ee_state()
        
        euler = pb.getEulerFromQuaternion(ori)
        
        observation.extend(jnt_pos)
        
        observation.extend(jnt_vel)

        observation.extend(list(pos))
        # observation.extend(list(euler))
        observation.extend(list(vel))
        # observation.extend(list(ee_vel[1]))

        return observation

    def ee_state(self, as_tuple=True):

        return self.get_ee_state(self._ee_index, as_tuple)

    def get_ee_wrench(self):
        '''
            End effector forces and torques
        '''

        jnt_poss, jnt_vels, jnt_reaction_forces, jnt_applied_torques = self.get_jnt_state(self._ft_sensor_jnt)

        return jnt_reaction_forces[:3], jnt_reaction_forces[3:]


    def inv_kin(self, ee_pos, ee_ori=None):

        cmd = self.get_ik(ee_idx=self._ee_index, ee_pos=ee_pos, ee_ori=ee_ori)
        #we delete the corresponding head ik from this
        return np.delete(cmd, 1, 0)


    def set_joint_state(self, joint_state):
        
        for k in range(len(self._movable_jnts)):
            
            pb.resetJointState(self._robot_id, self._movable_jnts[k], joint_state[k], physicsClientId=self._cid)

    def get_ee_pose(self):

        link_state = pb.getLinkState(self._robot_id, self._ee_index, physicsClientId=self._cid)
        
        ee_pos = np.asarray(link_state[0]) 
        
        ee_ori = np.asarray(link_state[1])

        return ee_pos, ee_ori

    def get_ee_velocity(self):

        link_state = pb.getLinkState(self._robot_id, self._ee_index, computeLinkVelocity = 1, physicsClientId=self._cid)

        ee_vel = np.asarray(link_state[6]) 
        
        ee_omg = np.asarray(link_state[7])

        return ee_vel, ee_omg 

    # def get_jnt_state(self):

    #     num_jnts = len(self._motor_indices)

    #     jnt_pos = []
        
    #     jnt_vel = []
        
    #     jnt_reaction_forces =  []
        
    #     jnt_applied_torque  = []

    #     for jnt_idx in range(len(self._motor_indices)):

    #         jnt_state = pb.getJointState(self._robot_id, self._motor_indices[jnt_idx], physicsClientId=self._cid)
            
    #         jnt_pos.append(jnt_state[0])
            
    #         jnt_vel.append(jnt_state[1])
            
    #         jnt_reaction_forces.append(jnt_state[2])
            
    #         jnt_applied_torque.append(jnt_state[3])

    #     return jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque

    def apply_action(self, motor_commands):

        for action in range (len(motor_commands)):

            motor = self._motor_indices[action]

            self.apply_ctrl(motor, motor_commands[action])
