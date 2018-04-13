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
from aml_rl_envs.sawyer.sawyer import Sawyer
from aml_rl_envs.utils.collect_demo import plot_demo
from aml_rl_envs.sawyer.config import SAWYER_ENV_CONFIG, SAWYER_CONFIG


class SawyerEnv(AMLRlEnv):
    
    def __init__(self, demo2follow=None, config=SAWYER_ENV_CONFIG):

        self._config = config

        self._action_repeat = config['action_repeat']

        self._goal_box = np.array([0.5, 1.,-0.35]) #x,y,z 

        self._demo2follow = demo2follow
        
        AMLRlEnv.__init__(self, config, set_gravity=True)

        self._reset()
        
        obs_dim = len(self.get_extended_observation())

        self.set_space_lims(obs_dim, 9, None, None, False)

        self._goal_ori = np.asarray(pb.getEulerFromQuaternion((-0.52021453, -0.49319602,  0.47898476, 0.50666373)))

        #facing sawyer, from left side
        hole1 = np.array([0., -0.725*0.15, 0.])
        hole2 = np.array([0., -0.425*0.15, 0.])
        hole3 = np.array([0., 0., 0.])
        hole4 = np.array([0., 0.375*0.15, 0.])
        hole5 = np.array([0., 0.775*0.15, 0.])

        self._hole_locs = [hole1, hole2, hole3, hole4, hole5]

    def _reset(self):

        self.setup_env()

        self._table_id = pb.loadURDF(os.path.join(self._urdf_root_path,"table.urdf"), useFixedBase=True, globalScaling=0.5, physicsClientId=self._cid)
        
        pb.resetBasePositionAndOrientation(self._table_id, [0.7, 0., 0.6], [0, 0, -0.707, 0.707], physicsClientId=self._cid)

        self._box_id = pb.loadURDF(os.path.join(self._urdf_root_path,"peg_hole.urdf"), useFixedBase=True, globalScaling = 0.15, physicsClientId=self._cid)
        
        pb.resetBasePositionAndOrientation(self._box_id, [0.65145, 0.015, 0.56], pb.getQuaternionFromEuler([1.57, 0., 1.57]), physicsClientId=self._cid) 
                        
        self._sawyer = Sawyer(config=SAWYER_CONFIG, cid=self._cid)

        self.simple_step()
        
        self._observation = self.get_extended_observation()

        return np.array(self._observation)


    def get_extended_observation(self):

         self._observation = self._sawyer.get_observation()

         gripper_state  = pb.getLinkState(self._sawyer._robot_id,self._sawyer._gripper_index, physicsClientId=self._cid)
         
         gripper_pos = gripper_state[0]
         
         gripper_ori = gripper_state[1]
         
         block_pos, block_ori = pb.getBasePositionAndOrientation(self._box_id, physicsClientId=self._cid)

         inv_gripper_pos, inv_gripper_ori = pb.invertTransform(gripper_pos, gripper_ori)
         
         block_pos_in_gripper, block_ori_in_gripper = pb.multiplyTransforms(inv_gripper_pos,inv_gripper_ori,block_pos,block_ori)
         
         projectedblock_pos2D =[block_pos_in_gripper[0],block_pos_in_gripper[1]]
         
         block_euler_in_gripper = pb.getEulerFromQuaternion(block_ori_in_gripper)

         #we return the relative x,y position and euler angle of block in gripper space
         block_in_gripper_ori_XYEulZ =[block_pos_in_gripper[0], block_pos_in_gripper[1], block_euler_in_gripper[2]]
         
         self._observation.extend(list(block_in_gripper_ori_XYEulZ))

         return self._observation
    
    def _step(self, action):
        
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
            
            self._sawyer.apply_action(action)
            
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

        state = pb.getLinkState(self._sawyer._robot_id, self._sawyer._ee_index, physicsClientId=self._cid)
        
        actualEndEffectorPos = state[0]
            
        blockPos, blockOrn=pb.getBasePositionAndOrientation(self._box_id, physicsClientId=self._cid)

        if (self._terminated or self._env_step_counter>self._max_steps):
            
            self._observation = self.get_extended_observation()
            
            return True

        if np.linalg.norm(np.asarray(blockPos) - self._goal_box) < 0.2:
                
            self._terminated = 1
            
            self._observation = self.get_extended_observation()
            
            return True

        return False
        
    def _reward(self):

        #rewards is height of target object

        block_pos,block_ori=pb.getBasePositionAndOrientation(self._box_id, physicsClientId=self._cid)
        
        closest_points     = pb.getClosestPoints(self._box_id, self._sawyer._robot_id, 1000, -1, self._sawyer._ee_index, physicsClientId=self._cid)

        ee_pos, ee_ori = self._sawyer.get_ee_pose()
        
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

    def reward(self, traj, end_id = 200, scale = [1,1]):
        '''
            Computing reward for the given (forward-simulated) trajectory
        '''

        def alignment_reward():

            # reference_vector = np.array([0,0,1]) # z-axis (direction of hole)

            ee_pos, ee_ori,_,_ = self._sawyer.ee_state()

            ori_diff_norm = np.linalg.norm(np.asarray(pb.getEulerFromQuaternion(ee_ori))-self._goal_ori)

            np.array([ [0.65145, 0.65145 + 0.15*0.55], 
                       [0.015,   0.015 - 0.15*1.90] ] )

            reward = -ori_diff_norm

            if (0.65145 < ee_pos[0] < 0.65145 + 0.15*0.55) and (0.015 < ee_pos[1] < 0.015 - 0.15*1.90):

                reward += 2.
            else:
                reward -= 2.

            # traj_end_vector = traj[-1, :] - traj[-end_id, :]
            # traj_end_vector = traj_end_vector/np.linalg.norm(traj_end_vector)
            # cos_angle = np.dot(traj_end_vector, reference_vector)

            return reward 


        # def completion_reward():

        #     # 0.6776011368  -0.1101703639   1.0655471412

        #     required_z_val = 1.06554

        #     # checking if the final position of the rolled-out trajectory reached the depth required for insertion
        #     if abs(traj[-1, 2] - required_z_val) < 0.2:
        #         reward = 10
        #     else:
        #         reward = -20

        #     return reward


        return scale[0]*alignment_reward() #+ scale[1]*completion_reward()

    def fwd_simulate(self, dmp, joint_space = False):
        """
        implement the dmp
        """
        # return np.random.randn(220,3)
        ee_traj = []


        for k in range(dmp.shape[0]):

            if joint_space:

                cmd = dmp[k, :]

            else:

                cmd = self._sawyer.inv_kin(ee_pos=dmp[k, :].tolist())

            self._sawyer.apply_action(cmd)

            ee_pos, ee_ori = self._sawyer.get_ee_pose()
            ee_traj.append(ee_pos)
            
            # import time
            # time.sleep(0.01)
            self.simple_step()

        block_pos, block_ori = pb.getBasePositionAndOrientation(self._box_id, physicsClientId=self._cid)

        # print "Block pos \t", np.asarray(block_pos)
        # print "EE pos\t", np.asarray(ee_pos)
            
        return np.asarray(ee_traj)
        
    def context(self):
        """
        Context is the top face of the box.

            Top face of box:
                # x : (0.7, 0.7 + 0.15*0.55)
                # y : (0.1, 0.1 - 0.15*1.90)
                # z : (0.62, 0.62 + 0.15*2.40)

                 0.69274812 -0.13284674  0.9552463
                 0.65145, 0.00966, 

                 0.6927 - 0.5*0.15*0.55 = 0.65145
                 -0.13284 + 0.5*0.15*1.9 = 0.00966
                 0.65145, 0.015
        """

        x = np.random.uniform(0.6700, 0.70000)#(0.65145, 0.65145 + 0.15*0.55)
        y = np.random.uniform(0.1, -0.21)#(0.015, 0.015 - 0.15*1.90)

        context = np.array([x,y])

        return context


    def execute_policy(self, w, s, show_demo=False):

        dmp = self._demo2follow(goal_offset=np.r_[w, 0]) #

        if show_demo:
            plot_demo(dmp, start_idx=0, life_time=4, cid=self._cid)

        traj = self.fwd_simulate(dmp)

        reward = self.reward(traj)
        
        return None, reward
