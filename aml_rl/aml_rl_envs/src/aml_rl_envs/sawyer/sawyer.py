import os
import sys
import copy
import math
import numpy as np
import pybullet as pb
from aml_rl_envs.aml_rl_robot import AMLRlRobot

class Sawyer(AMLRlRobot):

    def __init__(self, config):

        self._config = config

        self._ee_index = 24

        self._gripper_index = 7

        AMLRlRobot.__init__(self, config)

        self.reset()
 
    def reset(self):

        self._robot_id = pb.loadURDF(os.path.join(self._config['urdf_root_path'],"sawyer/sawyer2_with_gripper.urdf"), useFixedBase=True)

        pb.resetBasePositionAndOrientation(self._robot_id,[-0.100000,0.000000,0.070000],[0.000000,0.000000,0.000000,1.000000])
        
        self._jnt_postns=[ 0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048, 0.0, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200 ]
        
        self._movable_jnts = self.get_movable_joints()
        
        self._motor_names = []
        
        self._motor_indices = []
        
        for i in self._movable_jnts:
            
            jnt_info = pb.getJointInfo(self._robot_id, i)
            
            qIndex = jnt_info[3]
            
            if qIndex > -1:

                self._motor_names.append(str(jnt_info[1]))
                
                self._motor_indices.append(i)

        self.set_ctrl_mode()


    def get_action_dim(self):

        return 9 #position x,y,z and roll/pitch/yaw euler angles of end effector

    def get_obs_dim(self):

        return len(self.get_observation())

    def get_observation(self):
        
        observation = []
        
        ee_state = pb.getLinkState(self._robot_id, self._ee_index)
        
        ee_vel   = pb.getLinkState(self._robot_id, self._ee_index, computeLinkVelocity = 1)

        jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque = self.get_jnt_state()
        
        pos = ee_state[0]
        
        orn = ee_state[1]
        
        euler = pb.getEulerFromQuaternion(orn)
        
        observation.extend(jnt_pos)
        
        observation.extend(jnt_vel)

        observation.extend(list(pos))
        # observation.extend(list(euler))
        observation.extend(list(ee_vel[0]))
        # observation.extend(list(ee_vel[1]))

        return observation


    def set_joint_state(self, joint_state):
        
        for k in range(len(self._movable_jnts)):
            
            pb.resetJointState(self._robot_id, self._movable_jnts[k], joint_state[k])

    def get_ee_pose(self):

        link_state = pb.getLinkState(self._robot_id, self._ee_index)
        
        ee_pos = np.asarray(link_state[0]) 
        
        ee_ori = np.asarray(link_state[1])

        return ee_pos, ee_ori

    def get_ee_velocity(self):

        link_state = pb.getLinkState(self._robot_id, self._ee_index, computeLinkVelocity = 1)

        ee_vel = np.asarray(link_state[6]) 
        
        ee_omg = np.asarray(link_state[7])

        return ee_vel, ee_omg 

    def get_jnt_state(self):

        num_jnts = len(self._motor_indices)

        jnt_pos = []
        
        jnt_vel = []
        
        jnt_reaction_forces =  []
        
        jnt_applied_torque  = []

        for jnt_idx in range(len(self._motor_indices)):

            jnt_state = pb.getJointState(self._robot_id, self._motor_indices[jnt_idx])
            
            jnt_pos.append(jnt_state[0])
            
            jnt_vel.append(jnt_state[1])
            
            jnt_reaction_forces.append(jnt_state[2])
            
            jnt_applied_torque.append(jnt_state[3])

        return jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque

    def apply_action(self, motor_commands):

        for action in range (len(motor_commands)):

            motor = self._motor_indices[action]

            self.apply_ctrl(motor, motor_commands[action])
