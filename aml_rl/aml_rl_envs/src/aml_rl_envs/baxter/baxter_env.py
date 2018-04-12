import os
import gym
import time
import math
import random
import numpy as np
import pybullet as pb
from gym import spaces
from gym.utils import seeding
from aml_rl_envs.aml_rl_env import AMLRlEnv
from aml_rl_envs.baxter.baxter import Baxter
from aml_rl_envs.task.man_object import ManObject
from aml_rl_envs.utils.collect_demo import plot_demo
from aml_rl_envs.baxter.config import BAXTER_ENV_CONFIG, BAXTER_CONFIG


class BaxterEnv(AMLRlEnv):
    
    def __init__(self, demo2follow=None, config=BAXTER_ENV_CONFIG):

        self._config = config

        self._is_discrete = config['is_discrete']
        
        self._action_repeat = config['action_repeat']

        self._goal_box = np.array([0.5, 1.,-0.35]) #x,y,z 

        self._demo2follow = demo2follow
        
        AMLRlEnv.__init__(self, config, set_gravity=True)

        self.reset()
        
        obs_dim = len(self.get_extended_observation())

        self.set_space_lims(obs_dim, 18, None, None, self._is_discrete)

    def _reset(self):

        self.setup_env()

        pb.loadURDF(os.path.join(self._urdf_root_path,"plane.urdf"),[0,0,0], physicsClientId=self._cid)

        self._table_id = pb.loadURDF(os.path.join(self._urdf_root_path,"table.urdf"), useFixedBase=True, globalScaling=0.5, physicsClientId=self._cid)
        
        pb.resetBasePositionAndOrientation(self._table_id, [0.7, 0., 0.7], [0, 0, -0.707, 0.707], physicsClientId=self._cid)

        self._object = ManObject(cid=self._cid, urdf_root_path=self._config['urdf_root_path'], time_step=self._config['time_step'], 
                                  pos=[0.9, 0.1, 1.4], ori=[0, 0, -0.707, 0.707], scale=0.55, 
                                  use_fixed_Base = True, obj_type='sphere')
                        
        self._baxter = Baxter(config=BAXTER_CONFIG, cid=self._cid)
        
        if self._demo2follow is not None:
            #first joint position in the demo
            self._baxter.set_joint_state(self._demo2follow[0, :18])

        self.simple_step()
        
        self._observation = self.get_extended_observation()

        if self._renders and self._demo2follow is not None:
            
            plot_demo(trajectory=self._demo2follow, start_idx=18)

        return np.array(self._observation)


    def get_extended_observation(self):

         self._observation = self._baxter.get_observation()

         gripper_state  = pb.getLinkState(self._baxter._robot_id,self._baxter._gripper_index, physicsClientId=self._cid)
         
         gripper_pos = gripper_state[0]
         
         gripper_ori = gripper_state[1]
         
         block_pos, block_ori = self._object.get_curr_state(ori_type='quat', as_tuple=True)[:2]

         inv_gripper_pos, inv_gripper_ori = pb.invertTransform(gripper_pos, gripper_ori)
         
         block_pos_in_gripper, block_ori_in_gripper = pb.multiplyTransforms(inv_gripper_pos,inv_gripper_ori,block_pos,block_ori)
         
         projectedblock_pos2D =[block_pos_in_gripper[0],block_pos_in_gripper[1]]
         
         block_euler_in_gripper = pb.getEulerFromQuaternion(block_ori_in_gripper)

         #we return the relative x,y position and euler angle of block in gripper space
         block_in_gripper_ori_XYEulZ =[block_pos_in_gripper[0], block_pos_in_gripper[1], block_euler_in_gripper[2]]
         
         self._observation.extend(list(block_in_gripper_ori_XYEulZ))

         return self._observation
    
    def _step(self, action):
        
        if (self._is_discrete):

            jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque = self._baxter.get_jnt_state()

            dv = 0.005
            djoint_1 = [0, -dv, dv][action]
            djoint_2 = [0, -dv, dv][action]
            djoint_3 = [0, -dv, dv][action]
            djoint_4 = [0, -dv, dv][action]
            djoint_5 = [0, -dv, dv][action]
            djoint_6 = [0, -dv, dv][action]
            djoint_7 = [0, -dv, dv][action]
            djoint_8 = [0, -dv, dv][action]
            djoint_9 = [0, -dv, dv][action]

            real_action = [jnt_pos[0]+djoint_1, jnt_pos[1]+djoint_2, jnt_pos[2]+djoint_3,
                                        jnt_pos[3]+djoint_4, jnt_pos[4]+djoint_5, jnt_pos[5]+djoint_6,
                                        jnt_pos[6]+djoint_7, jnt_pos[7]+djoint_8, jnt_pos[8]+djoint_9]

        else:

            Kp1 = 1.; Kp2 = 1.; Kp3 = 1.
            Kp4 = 1.; Kp5 = 1.; Kp6 = 1.
            Kp7 = 1.; Kp8 = 1.; Kp9 = 1.

            joint_1  = action[0] * Kp1
            joint_2  = action[1] * Kp2
            joint_3  = action[2] * Kp3
            joint_4  = action[3] * Kp4
            joint_5  = action[4] * Kp5
            joint_6  = action[5] * Kp6
            joint_7  = action[6] * Kp7
            joint_8  = action[7] * Kp8
            joint_9  = action[8] * Kp9

            real_action = [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, joint_8, joint_9]

        return self.step2(real_action)
         
    def step2(self, action):
        
        for i in range(self._action_repeat):
            
            self._baxter.apply_action(action)
            
            pb.stepSimulation(physicsClientId=self._cid)
            
            if self.termination():
                
                break
            
            self._env_step_counter += 1
        
        if self._renders:
            
            time.sleep(self._time_step)
        
        self._observation = self.get_extended_observation()

        done = self.termination()
        
        npaction = np.array([action[3]]) #only penalize rotation until learning works well [action[0],action[1],action[3]])
        
        action_cost = np.linalg.norm(npaction)*10.

        reward = self._reward()-action_cost
        
        return np.array(self._observation), reward, done, {}

    def termination(self):

        state = pb.getLinkState(self._baxter._robot_id, self._baxter._ee_index, physicsClientId=self._cid)
        
        actualEndEffectorPos = state[0]
            
        blockPos, blockOrn=self._object.get_curr_state(ori_type='quat')[:2]

        if (self._terminated or self._env_step_counter>self._max_steps):
            
            self._observation = self.get_extended_observation()
            
            return True

        if np.linalg.norm(blockPos- self._goal_box) < 0.2:
                
            self._terminated = 1
            
            self._observation = self.get_extended_observation()
            
            return True

        return False
        
    def _reward(self):

        #rewards is height of target object

        blockPos, blockOrn=self._object.get_curr_state(ori_type='quat', as_tuple=True)[:2]
        
        closest_points     = pb.getClosestPoints(self._object._obj_id, self._baxter._robot_id, 1000, -1, self._baxter._ee_index, physicsClientId=self._cid)

        ee_pos, ee_ori = self._baxter.get_ee_pose()
        
        reward = -1.
        
        numPt = len(closest_points)

        distance_to_block = np.linalg.norm(np.asarray(block_pos) - ee_pos)

        if distance_to_block < 0.9:
            
            reward = reward + 5./distance_to_block

        distance_to_goal = np.linalg.norm(np.asarray(block_pos) - self._goal_box)
            
        if distance_to_goal < 0.9:
            
            reward = reward + 100/distance_to_goal
            
            print("successfully placed the block!!!")

        return reward

