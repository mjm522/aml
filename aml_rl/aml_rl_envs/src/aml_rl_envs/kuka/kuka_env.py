import os
import math
import time
import random
import numpy as np
import pybullet as pb
from aml_rl_envs.kuka.kuka import Kuka
from aml_rl_envs.aml_rl_env import AMLRlEnv
from aml_rl_envs.kuka.config import KUKA_ENV_CONFIG, KUKA_CONFIG

#Ref:https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/bullet

class KukaEnv(AMLRlEnv):

    def __init__(self, config=KUKA_ENV_CONFIG):

        self._config = config

        self._is_discrete = config['is_discrete']
        
        self._action_repeat = config['action_repeat']

        AMLRlEnv.__init__(self, config, set_gravity=True)

        self.reset()

        #timinglog = pb.startStateLogging(pb.STATE_LOGGING_PROFILE_TIMINGS, "kuka_timings.json")
        
        self._seed()
        
        obs_dim = len(self.get_extended_observation())

        self.set_space_lims(obs_dim, 7, None, None, self._is_discrete)
        

    def _reset(self):

        pb.loadURDF(os.path.join(self._urdf_root_path,"plane.urdf"),[0,0,-1])
        
        self._table_id = pb.loadURDF(os.path.join(self._urdf_root_path,"table.urdf"), useFixedBase=True)

        pb.resetBasePositionAndOrientation(self._table_id, [0.5,0.0, 0.0], [0.0,0.0,0.0,1.0])
        
        xpos = 0.55 +0.12*random.random()
        
        ypos = 0 +0.2*random.random()
        
        ang = 3.14*0.5+3.1415925438*random.random()
        
        orn = pb.getQuaternionFromEuler([0,0,ang])
        
        self._block_id =pb.loadURDF(os.path.join(self._urdf_root_path, "block.urdf"), xpos,ypos, 0.06,orn[0],orn[1],orn[2],orn[3])
                        
        self._kuka = Kuka(config=KUKA_CONFIG)
        
        self.simple_step()
        
        self._observation = self.get_extended_observation()
        
        return np.array(self._observation)


    def get_extended_observation(self):

         self._observation = self._kuka.get_observation()

         gripper_state  = pb.getLinkState(self._kuka._robot_id,self._kuka._gripper_index)
         
         gripper_pos = gripper_state[0]
         
         gripper_ori = gripper_state[1]
         
         block_pos, block_ori = pb.getBasePositionAndOrientation(self._block_id)

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
            
            dv = 0.005
            
            dx = [0,-dv,dv,0,0,0,0][action]
            
            dy = [0,0,0,-dv,dv,0,0][action]
            
            da = [0,0,0,0,0,-0.05,0.05][action]
            
            f = 0.3

            real_action = [dx,dy,-0.002,da,f]
        
        else:
 
            dv = 0.005
            
            dx = action[0] * dv
            
            dy = action[1] * dv
            
            da = action[2] * 0.05
            
            f = 0.3
            
            real_action = [dx,dy,-0.002,da,f]

        return self.step2( real_action)
         
    def step2(self, action):
        
        for i in range(self._action_repeat):
            
            self._kuka.apply_action(action)
            
            pb.stepSimulation()
            
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

    def _render(self, mode="rgb_array", close=False):
        
        if mode != "rgb_array":
            
            return np.array([])
        
        base_pos,orn = self._pb.getBasePositionAndOrientation(self._kuka._robot_id)
        
        view_matrix = self._pb.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=self._cam_dist,
                yaw=self._cam_yaw,
                pitch=self._cam_pitch,
                roll=0,
                upAxisIndex=2)
        
        proj_matrix = self._pb.computeProjectionMatrixFOV(
                fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
                nearVal=0.1, farVal=100.0)
        
        (_, _, px, _, _) = self._pb.getCameraImage(
                width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
                projectionMatrix=proj_matrix, renderer=self._pb.ER_BULLET_HARDWARE_OPENGL)
        
        rgb_array = np.array(px)
        
        rgb_array = rgb_array[:, :, :3]
        
        return rgb_array


    def termination(self):
        
        state = pb.getLinkState(self._kuka._robot_id, self._kuka._ee_index)
        
        actual_ee_pos = state[0]
            
        if (self._terminated or self._env_step_counter > self._max_steps):

            self._observation = self.get_extended_observation()
            
            return True
        
        max_dist = 0.005 
        
        closest_points = pb.getClosestPoints(self._kuka._tray_id, self._kuka._robot_id, max_dist)
         
        if (len(closest_points)):
            
            self._terminated = 1
        
            #start grasp and terminate
            finger_angle = 0.3
            
            for i in range (100):
                
                grasp_action = [0,0,0.0001,0,finger_angle]
                
                self._kuka.apply_action(grasp_action)
                
                pb.stepSimulation()
                
                finger_angle = finger_angle-(0.3/100.)
                
                if (finger_angle<0):
                    
                    finger_angle=0
        
            for i in range (1000):
                
                grasp_action = [0,0,0.001,0, finger_angle]
                
                self._kuka.apply_action(grasp_action)
                
                pb.stepSimulation()
                
                block_pos, block_ori =pb.getBasePositionAndOrientation(self._block_id)
                
                if (block_pos[2] > 0.23):

                    break
                
                state = pb.getLinkState(self._kuka._robot_id,self._kuka._ee_index)
                
                actual_ee_pos = state[0]
                
                if (actual_ee_pos[2]>0.5):
                    
                    break

            self._observation = self.get_extended_observation()
            
            return True
        
        return False
        
    def _reward(self):
        
        #rewards is height of target object
        block_pos,block_ori=pb.getBasePositionAndOrientation(self._block_id)
        
        closest_points = pb.getClosestPoints(self._block_id,self._kuka._robot_id,1000, -1, self._kuka._ee_index) 

        reward = -1000
        
        numPt = len(closest_points)

        if (numPt>0):
  
            reward = -closest_points[0][8]*10
        
        if (block_pos[2] >0.2):
            
            reward = reward+10000
            
            print("successfully grasped a block!!!")

        return reward
