import os
import gym
import time
import math
import random
import numpy as np
import pybullet as pb
from gym import spaces
from gym.utils import seeding
from aml_io.log_utils import aml_logging
from aml_rl_envs.aml_rl_env import AMLRlEnv
from aml_rl_envs.sawyer.sawyer import Sawyer
from aml_rl_envs.utils.collect_demo import plot_demo
from aml_rl_envs.sawyer.config import SAWYER_ENV_CONFIG, SAWYER_CONFIG

SAWYER_ENV_CONFIG['ctrl_type'] = 'torque'
SAWYER_CONFIG['ctrl_type'] = 'torque'

class SawyerEnv(AMLRlEnv):
    
    def __init__(self, demo2follow=None, config=SAWYER_ENV_CONFIG):

        self._config = config

        self._logger = aml_logging.get_logger(__name__)

        self._action_repeat = config['action_repeat']

        self._goal_box = np.array([0.5, 1.,-0.35]) #x,y,z 

        self._demo2follow = demo2follow
        
        AMLRlEnv.__init__(self, config, set_gravity=False)

        self._reset()
        
        obs_dim = len(self.get_extended_observation())

        self.set_space_lims(obs_dim, 9, None, None, False)

        self._goal_ori = np.asarray(pb.getEulerFromQuaternion((-0.52021453, -0.49319602,  0.47898476, 0.50666373)))

        #facing sawyer, from left side
        hole1 = np.array([0., -0.725*0.15, 0.])
        hole2 = np.array([0., -0.425*0.15, 0.])
        hole3 = np.array([0., 0., 0.0])
        hole4 = np.array([0., 0.375*0.15, 0.])
        hole5 = np.array([0., 0.775*0.15, 0.])

        self._hole_locs = [hole1, hole2, hole3, hole4, hole5]

    def _reset(self):

        self.setup_env()

        self._table_id = pb.loadURDF(os.path.join(self._urdf_root_path,"table.urdf"), useFixedBase=True, globalScaling=0.5, physicsClientId=self._cid)
        
        pb.resetBasePositionAndOrientation(self._table_id, [0.69028195, -0.08618135, -.08734368], [0, 0, -0.707, 0.707], physicsClientId=self._cid)

        # self._box_id = pb.loadURDF(os.path.join(self._urdf_root_path,"peg_hole/square_hole_vertical_18x22.urdf"), useFixedBase=True, globalScaling = 0.11, physicsClientId=self._cid)

        # pb.resetBasePositionAndOrientation(self._box_id, [0.69028195, -0.08618135, -.08734368], pb.getQuaternionFromEuler([0., 0., -3.14/2]), physicsClientId=self._cid) 

        self._box_id = pb.loadURDF('/home/br/gitlibs/trimesh/demos/old/demos.urdf', useFixedBase=True, globalScaling = 0.009, physicsClientId=self._cid)


        pb.resetBasePositionAndOrientation(self._box_id, [0.69028195, -0.08618135, .0], pb.getQuaternionFromEuler([3*np.pi/2, 0., 0.]), physicsClientId=self._cid) 
                        
        SAWYER_CONFIG['enable_force_torque_sensors'] = True

        initial_angle = [0.5775459,  0.35958105, -1.4546709,   1.94019824,  1.84663965,  1.72250488, 4.50212891]
    
        self._sawyer = Sawyer(config=SAWYER_CONFIG, cid=self._cid, jnt_pos=initial_angle)

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
            
        block_pos, block_orn=pb.getBasePositionAndOrientation(self._box_id, physicsClientId=self._cid)

        if (self._terminated or self._env_step_counter>self._max_steps):
            
            self._observation = self.get_extended_observation()
            
            return True

        if np.linalg.norm(np.asarray(block_pos) - self._goal_box) < 0.2:
                
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

    def reward(self, traj_reach, traj_insert, end_id = 200, scale = [0.003, 0.2, 2, 0.25]):
        '''
            Computing reward for the given (forward-simulated) trajectory
        '''

        def alignment_reward():

            # reference_vector = np.array([0,0,1]) # z-axis (direction of hole)

            ee_pos, ee_ori,_,_ = self._sawyer.ee_state()

            block_pos, block_orn=pb.getBasePositionAndOrientation(self._box_id, physicsClientId=self._cid)

            block_ori = np.asarray(pb.getEulerFromQuaternion(block_orn)) % np.pi
            ee_ori = (np.asarray(pb.getEulerFromQuaternion(ee_ori)) - np.array([[-3.08578398,  0.11448067, -3.1304143+1.57 ]])).squeeze() % np.pi

            # if block_ori[2] > 3.14:
            #     block_ori[2] = 3.14 - block_ori[2]

            # print "block ori \t", block_ori
            # print "ee_ori \t", ee_ori

            # raw_input()

            return -np.linalg.norm(ee_ori[2]-block_ori[2])


        def penalise_wrong_contacts(traj_data):

            penalty = 0

            # contact_list = traj_data['contact_details']

            # for i in range(len(contact_list)):

            #     if contact_list[i] is None:
            #         continue

            #     else:

            #         for contact_num in range(len(contact_list[i])):
            #             # penalise for any contact that is not with base of the box (-1)
            #             if contact_list[i][contact_num]['obj_link'] != -1:
            #                 penalty += 1
            #             else:
            #                 penalty -= 1

            return -penalty

        def check_final_ee_pos():

            block_pos, block_orn=pb.getBasePositionAndOrientation(self._box_id, physicsClientId=self._cid)

            final_ee_pos = traj_insert['ee_traj'][-1]

            r = -np.linalg.norm(final_ee_pos[:2]-block_pos[:2])

            if r > 0.1:

                r = -np.linalg.norm(final_ee_pos[2] - block_pos[2])


            return r 


            # reached_goal = False

            # final_contacts = traj_insert['contact_details'][-1]

            # if final_contacts is not None:

            #     for conts in range(len(final_contacts)):

            #         if final_contacts[conts]['obj_link'] == -1:
            #             reached_goal = True

            # reward = 10 if reached_goal else -2

            # traj 

            # return reward

        def closeness_reward():

            final_position = traj_reach['ee_traj'][-1]

            block_pos,block_ori=pb.getBasePositionAndOrientation(self._box_id, physicsClientId=self._cid)

            return -(np.linalg.norm(final_position[:2] - np.array([block_pos[0],  block_pos[1]])))

        def reaction_wrench_reward(traj):

            contact_details = traj['contact_details']

            forces_list = traj['ee_wrenches'][:,:3]
            torques_list = traj['ee_wrenches'][:,3:]

            for i in range(len(contact_details)):

                if contact_details[i] is not None:

                    curr_force  = forces_list[i,:]
                    curr_torque = torques_list[i,:]



        # def completion_reward():

        #     # 0.6776011368  -0.1101703639   1.0655471412

        #     required_z_val = 1.06554

        #     # checking if the final position of the rolled-out trajectory reached the depth required for insertion
        #     if abs(traj[-1, 2] - required_z_val) < 0.2:
        #         reward = 10
        #     else:
        #         reward = -20

        #     return reward

        # penalty_wrong_contact = scale[0]*(penalise_wrong_contacts(traj_reach) + penalise_wrong_contacts(traj_insert) )
        #0.003, 0.2, 2, 0.25
        penalty_final_ee_pos = 5*check_final_ee_pos()
        penalty_alignment = 3*alignment_reward()
        penalty_closeness = 0.1*closeness_reward()

        self._logger.debug("*******************************************************************")
        # self._logger.debug("penalty_wrong_contact \t %f"%(penalty_wrong_contact,))
        self._logger.debug("penalty_final_ee_pos \t %f"%(penalty_final_ee_pos,))
        self._logger.debug("penalty_alignment \t %f"%(penalty_alignment,))
        self._logger.debug("penalty_closeness \t %f"%(penalty_closeness,))
        self._logger.debug("*******************************************************************")

        # raw_input()

        tot_reward = penalty_final_ee_pos+\
                     penalty_alignment +\
                     penalty_closeness

        return  tot_reward

    def fwd_simulate(self, dmp, ee_ori=None, joint_space=False, Kp=None):
        """
        implement the dmp
        """
        ee_traj = []
        full_contacts_list = []
        ee_wrenches = []

        if ee_ori is None:
            goal_ori = (2.73469166e-02, 9.99530233e-01, 3.31521029e-04, 1.38329146e-02)
        else:
            goal_ori = pb.getQuaternionFromEuler(ee_ori)

        for k in range(dmp.shape[0]):

            if joint_space:

                cmd = dmp[k, :]

            else:
                # goal_ori = (-0.52021453, -0.49319602, 0.47898476, 0.50666373)
                # goal_ori = 
                # #in euler (3.1401728051502205, 0.027638219089217236, 3.0868671400979135)
                # print pb.getEulerFromQuaternion(goal_ori)
                # raw_input()
                # print "Goal pos \t",dmp[k, :].tolist()
                cmd = self._sawyer.inv_kin(ee_pos=dmp[k, :].tolist(), ee_ori=goal_ori)

            self._sawyer.apply_action(cmd, Kp)

            ee_pos, ee_ori = self._sawyer.get_ee_pose()

            ee_traj.append(ee_pos)

            full_contacts_list.append(self.get_contact_details())

            ee_wrenches.append(self._sawyer.get_ee_wrench())
            
            # time.sleep(0.1)
            self.simple_step()

        # block_pos, block_ori = pb.getBasePositionAndOrientation(self._box_id, physicsClientId=self._cid)
        # print "Block pos \t", np.asarray(block_pos)
        # print "EE pos\t", np.asarray(ee_pos)
            
        return { 'ee_traj':np.asarray(ee_traj), 'contact_details':full_contacts_list, 'ee_wrenches':np.asarray(ee_wrenches) }
        
    def context(self):
        """
        Context is the bottom base of the box.
        """

        s = np.random.uniform(-0.1, 0.1)

        box_ori = (0., 0., -3.14/2 + s)

        pb.resetBasePositionAndOrientation(self._box_id, [0.7076992,  -0.11391704,  0.61987786], pb.getQuaternionFromEuler(box_ori), physicsClientId=self._cid)

        block_pos, block_orn=pb.getBasePositionAndOrientation(self._box_id, physicsClientId=self._cid)

        block_orn = pb.getEulerFromQuaternion(block_orn)

        # return np.asarray(block_pos)
        return np.array([block_orn[2]])


    def execute_policy(self, w, s, show_demo=False):

        reach_end_offset=w[:3]
        peg_insert_end = np.r_[w[:2], w[3]]
        ee_ori_offset = np.array([0., 0., w[4]])

        # lin_jac = self._sawyer.jacobian()

        # Kp = np.ones(7) - np.dot(lin_jac.T, np.r_[w[5:8], w[5:8]])

        ee_ori = tuple(np.array([3.1401728051502205, 0.027638219089217236, 3.0868671400979135]) + ee_ori_offset)

        dmp_reach  = self._demo2follow(dmp_type='reach_hole', goal_offset=reach_end_offset)
        dmp_insert = self._demo2follow(dmp_type='insert', goal_offset=peg_insert_end, start_offset=reach_end_offset)

        if show_demo:
            plot_demo(dmp_reach, start_idx=0, life_time=4, cid=self._cid)
            plot_demo(dmp_insert, start_idx=0, life_time=4, cid=self._cid)

        traj_reach = self.fwd_simulate(dmp=dmp_reach, ee_ori=ee_ori)#, Kp=Kp)
        traj_insert = self.fwd_simulate(dmp=dmp_insert, ee_ori=ee_ori)#, Kp=Kp)

        reward = self.reward(traj_reach, traj_insert) #- 4*np.linalg.norm(Kp)
        
        return None, reward


    def get_contact_details(self):
        '''
            Get contact details of every contact when the peg is in contact with any part of the hole.
        '''

        full_details = pb.getContactPoints(bodyA=self._sawyer._robot_id, linkIndexA=19, 
                              bodyB=self._box_id, physicsClientId=self._cid)

        if len(full_details) > 0:

            contact_details = []

            for contact_id in range(len(full_details)):

                details = {}

                details['obj_link'] = full_details[contact_id][4]
                details['contact_pt'] = full_details[contact_id][6]
                details['contact_force'] = full_details[contact_id][9]

                contact_details.append(details)

            return contact_details

        return None