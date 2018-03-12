
import time
import numpy as np
import pybullet as pb
from hand import Hand

import os
import copy
import random
from man_object import ManObject
from os.path import exists, join

from aml_rl_envs.aml_rl_env import AMLRlEnv
from aml_rl_envs.config import urdf_root_path
from aml_rl_envs.utils.math_utils import skew
from aml_lfd.dmp.discrete_dmp import DiscreteDMP
from aml_lfd.dmp.config import discrete_dmp_config

from aml_rl_envs.utils.collect_demo import plot_demo, get_demo
from aml_rl_envs.hand.config import HAND_OBJ_CONFIG, HAND_CONFIG

class HandObstacleEnv(AMLRlEnv):


    def __init__(self, action_dim, demo2follow=None, 
                       action_high=None, action_low=None, 
                       randomize_box_ori=False, keep_obj_fixed = True, 
                       config=HAND_OBJ_CONFIG):

        self._config = config

        self._goal_block_pos = np.array([0, 0, 0.75]) #x,y,z
        
        self._goal_obj_ori = np.array([0.0, 0.00, -1.158])

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

        AMLRlEnv.__init__(self, config, set_gravity=False)

        self._reset(obj_base_fixed = keep_obj_fixed)

        obs_dim = len(self.get_extended_observation())

        self.set_space_lims(obs_dim, action_dim, action_high, action_low)


    def _reset(self, box_pos=[0.7, 0, 0.9], obj_base_fixed = True): #1.3

        self.setup_env()

        scale = 1

        if self._randomize_box_ori:
            
            box_ori = pb.getQuaternionFromEuler([0., 0., np.random.uniform(-0.08*np.pi, 0.08*np.pi)])
        
        else:
            
            box_ori = [0.,0.,0.,1]

        self._world_id = pb.loadURDF(join(self._urdf_root_path,"plane.urdf"))
        
        self._table_id = pb.loadURDF(join(self._urdf_root_path, "table.urdf"), useFixedBase=True)
        
        self._obstacle_id  = pb.loadURDF(join(self._urdf_root_path, "topple_block.urdf"), useFixedBase=True)

        pb.resetBasePositionAndOrientation(self._world_id, [0., 0., -0.5], [0.,0.,0.,1])

        pb.resetBasePositionAndOrientation(self._table_id, [0., 0., 0.75], [0.,0.,0.,1])

        obs_pos=[0, 0, 0.815]
        
        obs_ori=[0.,0.,0.,1] 
        
        pb.resetBasePositionAndOrientation(self._obstacle_id, obs_pos, obs_ori)
 
        self._object = ManObject(config=self._config, pos=box_pos, ori=box_ori, scale=scale, use_fixed_Base = obj_base_fixed)
        
        base_hand_pos  = [0.4, 0., 2]

        base_hand_ori  = [0,1,0,0]

        hand_j_pos = [0, np.pi/2, 0.0, 0.000, 0, 1, -1.2, 0.00]

        self._hand = Hand(config=HAND_CONFIG, pos=base_hand_pos, ori=base_hand_ori, j_pos=hand_j_pos, hand_choice = 'pincer')

        self._num_fingers = self._hand._num_fingers

        self._num_joints_finger = self._hand._num_joints_per_finger

        self.set_friction_properties()

        # self._observation = self.get_extended_observation()

        if self._renders and self._demo2follow is not None:
            plot_demo(trajectory=self._demo2follow[0])
            plot_demo(trajectory=self._demo2follow[1])
            plot_demo(trajectory=self._demo2follow[2])


        return np.array(self._observation)


    def set_friction_properties(self):

        pb.setPhysicsEngineParameter(restitutionVelocityThreshold=0.2)

        pb.changeDynamics(self._table_id,-1,  lateralFriction=1, spinningFriction=1., rollingFriction=1., restitution=0.6)

        self._hand.set_friction_properties(lf=1., sf=1., rf=1., r=0.)

        self._object.set_friction_properties(lf=0.01, sf=0., rf=0., r=0.)


    def get_hand_limits(self):

        return self._hand.get_finger_limits()

    def get_hand_joint_state(self):

        jnt_poss, jnt_vels, jnt_reaction_forces, jnt_applied_torques = self._hand.get_jnt_states()

        return {'pos_js':jnt_poss,
                'vel_js':jnt_vels,
                'rea_force_js':jnt_reaction_forces,
                'apl_torque_js':jnt_applied_torques}


    def get_hand_ee_state(self):

        ee_poss, ee_oris, ee_vels, ee_omgs = self._hand.get_ee_states(as_tuple=True)

        return {'pos_ee': ee_poss,
                'vel_ee': ee_vels}


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


    def get_contact_points_robot_object(self):
        #only report contact points that involve male and female, the table (environment contact) is neglected

        contact_points = []

        for fin_no in range(self._hand._num_fingers):

            contact_data = pb.getContactPoints(bodyA=self._hand._robot_id, linkIndexA=self._hand._ee_indexs[fin_no], bodyB=self._object._obj_id, linkIndexB=0)

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

    def get_contact_points_object_obstacle(self):
        #only report contact points that involve block and obstacle, the table (environment contact) is neglected

        contact_data = pb.getContactPoints(bodyA=self._object._obj_id, bodyB=self._obstacle_id)

        #contact points on male
        cp_on_block = []
        #contact points on obstacle
        cp_on_obstacle = []
        #contact normals on obstacle
        cn_on_obstacle = []
        #contact distance 
        c_dist = []
        #contact normal force
        cn_force = []

        for contact in contact_data:

            cp_on_block.append(contact[5])
            cp_on_obstacle.append(contact[6])
            cn_on_obstacle.append(contact[7])
            c_dist.append(contact[8])
            cn_force.append(contact[9])

        return {'cp_on_block':cp_on_block,
                'cp_on_obstacle':cp_on_obstacle,
                'cn_on_obstacle':cn_on_obstacle,
                'c_dist':c_dist,
                'cn_force':cn_force}

    def get_contact_points_object_table(self):
        #only report contact points that involve block and obstacle, the table (environment contact) is neglected

        contact_data = pb.getContactPoints(bodyA=self._object._obj_id, bodyB=self._table_id)

        #contact points on male
        cp_on_block = []
        #contact points on obstacle
        cp_on_table = []
        #contact normals on obstacle
        cn_on_table = []
        #contact distance 
        c_dist = []
        #contact normal force
        cn_force = []

        for contact in contact_data:

            cp_on_block.append(contact[5])
            cp_on_table.append(contact[6])
            cn_on_table.append(contact[7])
            c_dist.append(contact[8])
            cn_force.append(contact[9])

        return {'cp_on_block':cp_on_block,
                'cp_on_table':cp_on_table,
                'cn_on_table':cp_on_table,
                'c_dist':c_dist,
                'cn_force':cn_force}



    def get_closest_points(self, eps_dist=0.02):
        #only report contact points that involve male and female, the table (environment contact) is neglected

        closest_points = []

        for fin_no in range(self._hand._num_fingers):

            contact_data = pb.getContactPoints(bodyA=self._hand._robot_id, linkIndexA=self._hand._ee_indexs[fin_no], bodyB=self._object._obj_id, linkIndexB=0)

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


    def get_ik(self, finger_idx, ee_pos, ee_ori=None):

        return np.asarray(self._hand.inv_kin(finger_idx=finger_idx,ee_pos=ee_pos, ee_ori=ee_ori))


    def get_object_mass_matrix(self):

        return self._object.get_mass_matrix()

    def get_obj_kin_state(self, ori_type='eul'):
        #get_obj_curr_state =>old name

        return self._object.get_curr_state(ori_type=ori_type)

    def get_obj_dyn_state(self):

        return self._object.get_jnt_state()


    def get_robot_curr_state(self):

        js_state = self.get_hand_joint_state()
        ee_state = self.get_hand_ee_state()

        return dict(js_state, **ee_state)


    def transfer_point_from_world_to_obj(self, point, obj_pos=None, obj_ori=None):

        if isinstance(point, np.ndarray) or isinstance(point, list):
            point = tuple(point)

        if (obj_pos is None) or (obj_ori is None):
            obj_pos, obj_ori = self._object.get_base_pose()

        inv_obj_pos, inv_obj_ori = pb.invertTransform(obj_pos, obj_ori)

        point_pos_in_obj, point_ori_in_obj = pb.multiplyTransforms(inv_obj_pos, inv_obj_ori, point, (0.0, 0.0, 0.0, 1.0))

        return np.asarray(point_pos_in_obj), np.asarray(point_ori_in_obj)

    def transfer_point_from_obj_to_world(self, point, obj_pos=None, obj_ori=None):

        if isinstance(point, np.ndarray) or isinstance(point, list):
            point = tuple(point)

        if (obj_pos is None) or (obj_ori is None):
            obj_pos, obj_ori = self._object.get_base_pose()

        point_pos_in_world, point_ori_in_world = pb.multiplyTransforms(obj_pos, obj_ori, point, (0.0, 0.0, 0.0, 1.0))

        return np.asarray(point_pos_in_world), np.asarray(point_ori_in_world)


    def compute_grasp_map(self, finger_idx, x_des=None):

        contact_points = self.get_contact_points()

        #point contact with friction, shastry page, 219
        wrench_basis = np.vstack([np.diag(np.ones(3)), np.zeros([3, 3])])

        #this is a huge matrix, four four fingers, it will be 6x12
        grasp_map = np.zeros([6, 3*self._num_fingers])

        ee_poss, ee_oris, ee_vels, ee_omgs = self._hand.get_ee_states(as_tuple=True)

        obj_pos, obj_ori = self._object.get_base_pose()

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

            obj_pos, obj_ori = self._object.get_base_pose()

            inv_ee_pos,inv_ee_ori = pb.invertTransform(ee_pos, ee_ori)

            obj_pos_ee, obj_ori_ee = pb.multiplyTransforms(inv_ee_pos, inv_ee_ori, obj_pos, obj_ori)

            obj_euler_ee = pb.getEulerFromQuaternion(obj_ori_ee)

            #we return the relative x,y position and euler angle of block in gripper space
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

        if self._demo2follow is None:
            raise Exception("No dmp given...")

        return self.run_dmp(phase_start1=action[0], start_offset1=np.zeros(3), goal_offset1=np.zeros(3), Kp1=np.ones(3),
                            phase_start2=action[1], start_offset2=np.zeros(3), goal_offset2=np.zeros(3), Kp2=np.ones(3),
                            phase_start3=action[2], start_offset3=np.zeros(3), goal_offset3=np.zeros(3), Kp3=np.ones(3))

        # return self.run_dmp(start_offset1=action[:3], goal_offset1=action[3:6], Kp1=action[6:9],
        #                     start_offset2=action[9:12], goal_offset2=action[12:15], Kp2=action[15:])

     
    def step2(self, action, Kp):

        self._hand.apply_action(action, Kp)

        pb.stepSimulation()

        done = self._termination()

        self._env_step_counter += 1

        if self._renders:
            
            time.sleep(self._time_step)

        self._observation = self.get_extended_observation()

        #add the contact penalization here

        reward = self._reward()

        return np.array(self._observation), reward, done, {}


    def _termination(self):

        obj_pos, obj_ori = self._object.get_base_pose()

        if (self._terminated or self._env_step_counter>self._max_steps):
            self._observation = self.get_extended_observation()
            return True

        if np.linalg.norm(np.asarray(pb.getEulerFromQuaternion(obj_ori)) - self._goal_obj_ori) < 0.06:#(actualEndEffectorPos[2] <= -0.43):
            self._terminated = 1
            self._observation = self.get_extended_observation()
            return True

        return False
        
    def _reward(self):

        reward = -1.

        ee_poss, ee_oris, ee_vels, ee_omgs = self._hand.get_ee_states()

        for fin_no in range(self._hand._num_fingers):

            #rewards is height of target object
            obj_pos, obj_ori = self._object.get_base_pose()
            
            distance_to_obj = np.linalg.norm(np.asarray(obj_pos) - ee_poss[fin_no])

            if distance_to_obj < 0.9:
                reward +=  2./distance_to_obj

            distance_to_goal = np.linalg.norm(np.asarray(pb.getEulerFromQuaternion(obj_ori)) - self._goal_obj_ori)
              
            if distance_to_goal < 0.9:
                reward = reward + 10/distance_to_goal

                if distance_to_goal < 0.06:
                    reward += 10/distance_to_goal
                    print("successfully rotated the block!!!")

        return reward

if __name__ == '__main__':


    env = HandObstacleEnv(renders=True, action_dim=6, randomize_box_ori=False, keep_obj_fixed=False, ctrlType='pos')

    cmd = np.array([0.,0., -2.35619])

    while True:

        # env._hand.set_joint_state(finger_idx=0, jnt_pos=cmd)

        pb.stepSimulation()

        # cmd += np.array([0.0, 0.00, 0.001])

        # # print env.get_contact_points()
        # # env.get_point_obj_frame()

        # raw_input(cmd)
        print env.get_object_obstacle_contacts()