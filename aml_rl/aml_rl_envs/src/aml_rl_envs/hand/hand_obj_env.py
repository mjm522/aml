
import time
import numpy as np
import pybullet as pb


import os
import copy
import random

from os.path import exists, join

from aml_rl_envs.hand.hand import Hand

from aml_rl_envs.aml_rl_env import AMLRlEnv
from aml_rl_envs.config import urdf_root_path
from aml_rl_envs.utils.math_utils import skew

from aml_lfd.dmp.discrete_dmp import DiscreteDMP
from aml_rl_envs.task.man_object import ManObject
from aml_lfd.dmp.config import discrete_dmp_config
from aml_rl_envs.config import AML_RL_ROBOT_CONFIG

from aml_rl_envs.utils.collect_demo import plot_demo, get_demo
from aml_rl_envs.hand.config import HAND_OBJ_CONFIG, HAND_CONFIG

class HandObjEnv(AMLRlEnv):

    def __init__(self,  action_dim,  demo2follow=None, 
                        action_high=None, action_low=None, 
                        randomize_box_ori=True, keep_obj_fixed = True, 
                        config=HAND_OBJ_CONFIG, hand_choice='four_finger',set_gravity=True):

        self._goal_block_pos = np.array([0, 0, 0.]) #x,y,z
        
        self._goal_obj_ori = np.array([0.0, 0.0, 2.45]) #1.97831951

        self._randomize_box_ori = randomize_box_ori

        self._demo2follow = demo2follow

        if self._demo2follow is not None:

            self._dmp1 = DiscreteDMP(config=discrete_dmp_config)
            self._dmp1.load_demo_trajectory(self._demo2follow[0][:, 8:11])
            self._dmp1.train()
            self._dmp1_config = copy.deepcopy(discrete_dmp_config)

            self._dmp2 = DiscreteDMP(config=discrete_dmp_config)
            self._dmp2.load_demo_trajectory(self._demo2follow[1][:, 8:11])
            self._dmp2.train()
            self._dmp2_config = copy.deepcopy(discrete_dmp_config)

            self._dmp3 = DiscreteDMP(config=discrete_dmp_config)
            self._dmp3.load_demo_trajectory(self._demo2follow[2][:, 8:11])
            self._dmp3.train()
            self._dmp3_config = copy.deepcopy(discrete_dmp_config)

        self._config = config

        AMLRlEnv.__init__(self, config, set_gravity=set_gravity)

        self._reset(obj_base_fixed = keep_obj_fixed, hand_choice=hand_choice)

        obs_dim = len(self.get_extended_observation())

        if HAND_OBJ_CONFIG['ctrl_type'] ==  'pos':

            hand_info = self._hand.get_joint_limits()

            action_high = hand_info['upper']

            action_low = hand_info['lower']


        self.set_space_lims(obs_dim, action_dim, action_high, action_low)

        self._seed()


    def _reset(self, box_pos=[0, 0, 1.4 ], obj_base_fixed = True, hand_choice='four_finger'):

        self.setup_env()

        scale = 1.2

        if self._randomize_box_ori:
            
            box_ori = pb.getQuaternionFromEuler([0., 0., np.random.uniform(-0.08*np.pi, 0.08*np.pi)])
        
        else:
            
            box_ori = [0.,0.,0.,1]

        self._world_id = pb.loadURDF(join(self._urdf_root_path,"plane.urdf"), physicsClientId=self._cid)
        
        self._table_id = pb.loadURDF(join(self._urdf_root_path, "table.urdf"), useFixedBase=True, physicsClientId=self._cid)

        pb.resetBasePositionAndOrientation(self._world_id, [0., 0., -0.5], [0.,0.,0.,1], physicsClientId=self._cid)
        
        pb.resetBasePositionAndOrientation(self._table_id, [0., 0., 0.5], [0.,0.,0.,1], physicsClientId=self._cid)

        self._object = ManObject(cid=self._cid, urdf_root_path=self._config['urdf_root_path'], time_step=self._config['time_step'], 
                                  pos=box_pos, ori=box_ori, scale=scale, 
                                  use_fixed_Base = obj_base_fixed, obj_type='cyl')
        
        base_hand_pos  = [0., 0., 0.7]
        
        base_hand_ori  = pb.getQuaternionFromEuler([0., 0., 0.])

        hand_j_pos = [0.014906321431778925, 0.176, -1.900697592866243, 0.0]#, 0.01464389158008051, 0.7442819989476841, -1.6298398854594098, 0.0]

        self._hand = Hand(cid=self._cid, config=HAND_CONFIG, pos=base_hand_pos, ori=base_hand_ori, j_pos=hand_j_pos, hand_choice=hand_choice) #

        self._num_fingers = self._hand._num_fingers
        
        self.set_friction_properties()

        self._env_step_counter = 0
        
        self.simple_step()

        return np.array(self._observation)


    def set_friction_properties(self):

        pb.setPhysicsEngineParameter(restitutionVelocityThreshold=0.2, physicsClientId=self._cid)

        pb.changeDynamics(self._table_id,-1,  lateralFriction=1, spinningFriction=1., rollingFriction=1., restitution=0.6, physicsClientId=self._cid)

        self._hand.set_friction_properties(lf=1., sf=1., rf=1., r=0.)

        self._object.set_friction_properties(lf=0.01, sf=0., rf=0., r=0.)

        
    def get_hand_limits(self):

        return self._hand.get_finger_limits()

    def get_hand_joint_state(self):

        jnt_poss, jnt_vels, jnt_reaction_forces, jnt_applied_torques = self._hand.get_jnt_states()

        return {'pos':jnt_poss,
                'vel':jnt_vels,
                'rea_force':jnt_reaction_forces,
                'apl_torque':jnt_applied_torques}


    def get_hand_ee_state(self):

        ee_poss, ee_oris, ee_vels, ee_omgs = self._hand.get_ee_states(as_tuple=True)

        return {'pos': ee_poss,
                'vel': ee_vels}


    def get_finger_jac(self, finger_idx, j_poss, local_point=None):
        
        return self._hand.get_finger_jacobian(finger_idx=finger_idx, jnt_poss=j_poss, local_point=local_point)


    def update_dmp_params(self, dmp_no, phase_start=1., speed=1., goal_offset=np.array([0., 0., 0.]), start_offset=np.array([0., 0., 0.]), external_force=None):

        if self._demo2follow is None:
            return
        
        if dmp_no == 1:
            
            dmp    = self._dmp1
            
            config = self._dmp1_config
        
        elif dmp_no == 2:
            
            dmp    = self._dmp2
            
            config = self._dmp2_config
        
        elif dmp_no == 3:
            
            dmp    = self._dmp3
            
            config = self._dmp3_config

        config['y0'] = dmpb._traj_data[0, 1:] + start_offset
        
        config['dy'] = np.array([0., 0., 0.])
        
        config['goals'] = dmpb._traj_data[-1, 1:] + goal_offset
        
        config['tau'] = 1./speed
        
        config['phase_start'] = phase_start

        if external_force is None:
            
            external_force = np.array([0.,0.,0.,0.])
            
            config['type'] = 1
        
        else:
            
            config['type'] = 3

        config['extForce'] = external_force

        new_dmp_traj = dmpb.generate_trajectory(config=config)

        return new_dmp_traj


    def get_contact_points(self):
        #only report contact points that involve male and female, the table (environment contact) is neglected

        contact_points = []

        for fin_no in range(self._hand._num_fingers):

            contact_data = pb.getContactPoints(bodyA=self._hand._robot_id, linkIndexA=self._hand._ee_indexs[fin_no], 
                                               bodyB=self._object._obj_id, linkIndexB=0, physicsClientId=self._cid)

            #contact points on male
            cp_on_finger = []
            #contact points on female
            cp_on_block = []
            #contact normals on female
            cn_on_block= []
            #contact distance 
            c_dist = []
            #contact normal force
            cn_force = []

            for contact in contact_data:

                cp_on_finger.append(contact[5])
                
                cp_on_block.append(contact[6])
                
                cn_on_block.append(contact[7])
                
                c_dist.append(contact[8])
                
                cn_force.append(contact[9])

            contact_points.append({'cp_on_finger':cp_on_finger,
                                  'cp_on_block':cp_on_block,
                                  'cn_on_block':cn_on_block,
                                  'c_dist':c_dist,
                                  'cn_force':cn_force})

        return contact_points


    def get_closest_points(self, eps_dist=0.02):
        #only report contact points that involve male and female, the table (environment contact) is neglected

        closest_points = []

        for fin_no in range(self._hand._num_fingers):

            contact_data = pb.getContactPoints(bodyA=self._hand._robot_id, linkIndexA=self._hand._ee_indexs[fin_no], 
                                               bodyB=self._object._obj_id, linkIndexB=0, physicsClientId=self._cid)

            #contact points on male
            cp_on_finger = []
            #contact points on female
            cp_on_block = []
            #contact normals on female
            cn_on_block= []
            #contact distance 
            c_dist = []
            #contact normal force
            cn_force = []

            for contact in contact_data:

                cp_on_finger.append(contact[5])
                
                cp_on_block.append(contact[6])
                
                c_dist.append(contact[8])
                
                cn_force.append(contact[9])

            closest_points.append({'cp_on_finger':cp_on_finger,
                                  'cp_on_block':cp_on_block,
                                  'c_dist':c_dist,
                                  'cn_force':cn_force})

        return closest_points


    def get_ik(self, finger_idx, ee_pos):

        return np.asarray(self._hand.inv_kin(finger_idx=finger_idx,ee_pos=ee_pos))


    def get_object_mass_matrix(self):

        return self._object.get_mass_matrix()

    def get_obj_curr_state(self, ori_type='eul'):

        return self._object.get_curr_state(ori_type=ori_type)


    def transfer_point_from_world_to_obj(self, point):

        if isinstance(point, np.ndarray) or isinstance(point, list):
            point = tuple(point)

        # obj_pos, obj_ori = self._object.get_base_pose()

        obj_pos, obj_ori = self._object.get_curr_state(ori_type='quat', as_tuple=True)[:2]

        inv_obj_pos, inv_obj_ori = pb.invertTransform(obj_pos, obj_ori)

        point_pos_in_obj, point_ori_in_obj = pb.multiplyTransforms(inv_obj_pos, inv_obj_ori, point, (0.0, 0.0, 0.0, 1.0))

        return np.asarray(point_pos_in_obj), np.asarray(point_ori_in_obj)

    def transfer_point_from_obj_to_world(self, point):

        if isinstance(point, np.ndarray) or isinstance(point, list):
            point = tuple(point)

        # obj_pos, obj_ori = self._object.get_base_pose()
        obj_pos, obj_ori = self._object.get_curr_state(ori_type='quat', as_tuple=True)[:2]

        point_pos_in_world, point_ori_in_world = pb.multiplyTransforms(obj_pos, obj_ori, point, (0.0, 0.0, 0.0, 1.0))

        return np.asarray(point_pos_in_world), np.asarray(point_ori_in_world)


    def compute_grasp_map(self, finger_idx, x_des=None):

        contact_points = self.get_contact_points()

        #point contact with friction, shastry page, 219
        wrench_basis = np.vstack([np.diag(np.ones(3)), np.zeros([3, 3])])

        #this is a huge matrix, four four fingers, it will be 6x12
        grasp_map = np.zeros([6, 3*self._num_fingers])

        ee_poss, ee_oris, ee_vels, ee_omgs = self._hand.get_ee_states(as_tuple=True)

        # obj_pos, obj_ori = self._object.get_base_pose()
        obj_pos, obj_ori = self._object.get_curr_state(ori_type='quat', as_tuple=True)[:2]

        dummy_contact_point_list = [tuple(np.asarray(obj_pos) + np.array([-0.34, 0., 0.])), tuple(np.asarray(obj_pos) + np.array([0.34, 0., 0.])) ]
        
        for k in range(self._hand._num_fingers):

            if k == finger_idx:
                continue

            contact_point = contact_points[k]['cp_on_finger']

            if contact_point:

                contact_point = contact_point[0]

                ee_ori =  ee_oris[k]

                Ree_W = np.asarray(pb.getMatrixFromQuaternion(ee_ori)).reshape(3,3)

                Ree_W = np.eye(3)

                skew_rot = np.dot(skew(np.asarray(contact_point) - np.asarray(obj_pos)), Ree_W)

                grasp_map[:, 3*k:3*k+3] = np.vstack([Ree_W, skew_rot])

        return grasp_map

    def compute_all_contact_forces_vector(self):
        
        all_contact_force_vector  = np.zeros(3*self._num_fingers)

        contact_points = self.get_contact_points()

        for k in range(self._num_fingers):

            if contact_points[k]['cp_on_finger']:

                all_contact_force_vector[3*k:3*k+3]  = contact_points[k]['cn_force'][0]*np.asarray(contact_points[k]['cn_on_block'][0])

        return all_contact_force_vector


    def get_extended_observation(self):

        self._observation = []

        self._observation.extend(self._hand.get_observation())

        ee_poss, ee_oris, ee_vels, ee_omgs = self._hand.get_ee_states(as_tuple=True)

        for fin_no in range(self._num_fingers):
            
            ee_pos = ee_poss[fin_no]
            ee_ori = ee_oris[fin_no]

            # obj_pos, obj_ori = self._object.get_base_pose()

            obj_pos, obj_ori = self._object.get_curr_state(ori_type='quat', as_tuple=True)[:2]

            inv_ee_pos,inv_ee_ori = pb.invertTransform(ee_pos, ee_ori)

            obj_pos_ee, obj_ori_ee = pb.multiplyTransforms(inv_ee_pos, inv_ee_ori, obj_pos, obj_ori)

            obj_euler_ee = pb.getEulerFromQuaternion(obj_ori_ee)

            obj_ee_pose = [obj_pos_ee[0],obj_pos_ee[1], obj_pos_ee[2], obj_euler_ee[0], obj_euler_ee[1], obj_euler_ee[2]]

            self._observation.extend(list(obj_ee_pose))

        return self._observation


    def run_dmp(self, phase_start1, start_offset1, goal_offset1, Kp1, 
                      phase_start2, start_offset2, goal_offset2, Kp2,
                      phase_start3, start_offset3, goal_offset3, Kp3):

        new_dmp_traj1 = self.update_dmp_params(dmp_no=1, phase_start=phase_start1, start_offset=start_offset1, goal_offset=goal_offset1)
        new_dmp_traj2 = self.update_dmp_params(dmp_no=2, phase_start=phase_start2, start_offset=start_offset2, goal_offset=goal_offset2)
        new_dmp_traj3 = self.update_dmp_params(dmp_no=3, phase_start=phase_start3, start_offset=start_offset3, goal_offset=goal_offset3)

        for k in range(0, max(new_dmp_traj1.shape[0], new_dmp_traj2.shape[0], new_dmp_traj3.shape[0])):
            #skipping indices to move faster

            if k >= new_dmp_traj1.shape[0]:
                
                idx1 = -1
            
            else:
                
                idx1 = k

            if k >= new_dmp_traj2.shape[0]:
                
                idx2 = -1
            
            else:
                
                idx2 = k

            if k >= new_dmp_traj2.shape[0]:
                
                idx3 = -1
            
            else:
                
                idx3 = k

            action1=self._finger1.inv_kin(new_dmp_traj1[idx1, 1:])
            
            action2=self._finger2.inv_kin(new_dmp_traj2[idx2, 1:])
            
            action3=self._finger2.inv_kin(new_dmp_traj2[idx3, 1:])

            state, reward, done, info = self.step2(action1=action1, 
                                                   action2=action2,
                                                   action3=action3, 
                                                   Kp1=Kp1, Kp2=Kp2, Kp3=Kp3)

            if done:

                break

        return state, reward, done, info

    def _step(self, action):

        return self.step2(action)

     
    def step2(self, action, Kp=None):

        self._hand.apply_jnt_ctrl(action)

        self.simple_step()

        done = self.termination()

        self._env_step_counter += 1

        if self._renders:
            time.sleep(self._time_step)

        self._observation = self.get_extended_observation()

        #add the contact penalization here

        reward = self._reward()

        return np.array(self._observation), reward, done, {}

    
    def termination(self):

        # obj_pos, obj_ori = self._object.get_base_pose()

        obj_pos, obj_ori = self._object.get_curr_state(ori_type='quat', as_tuple=True)[:2]

        if (self._terminated or self._env_step_counter>self._max_steps):
            self._observation = self.get_extended_observation()
            return True

        if np.linalg.norm(np.asarray(pb.getEulerFromQuaternion(obj_ori)) - self._goal_obj_ori) < 0.01:
            self._terminated = 1
            self._observation = self.get_extended_observation()

            print "here"
            return True

        return False
        
    def _reward(self):

        reward = -1.

        ee_poss, ee_oris, ee_vels, ee_omgs = self._hand.get_ee_states()

        for fin_no in range(self._hand._num_fingers):

            #rewards is height of target object
            # obj_pos, obj_ori = self._object.get_base_pose()
            obj_pos, obj_ori = self._object.get_curr_state(ori_type='quat', as_tuple=True)[:2]
            
            # distance_to_obj = np.linalg.norm(np.asarray(obj_pos) - ee_poss[fin_no])

            # if distance_to_obj < 0.9:
                
            #     reward +=  2./distance_to_obj

            distance_to_goal = np.linalg.norm(np.asarray(pb.getEulerFromQuaternion(obj_ori)) - self._goal_obj_ori)
              
            if distance_to_goal < 0.1:
                
                reward = reward + 10/distance_to_goal

                if distance_to_goal < 0.06:
                    
                    reward += 10/distance_to_goal
                    
                    print("successfully rotated the block!!!")

        return reward
