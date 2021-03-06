import os
import copy
import math
import numpy as np
import pybullet as pb
from os.path import exists, join
from aml_rl_envs.config import urdf_root_path
from aml_rl_envs.bullet_visualizer import setup_bullet_visualizer


class ManObject():

    def __init__(self, cid, urdf_root_path=urdf_root_path, time_step=0.01,
                       pos = [1.75, 0., 1.5], ori = [0., 0., 0., 1], 
                       j_pos=[0.0, 0.0], scale=1., use_fixed_Base=True, 
                       obj_type='cyl', render=False):

        self._cid = cid

        if render:

            self._cid = setup_bullet_visualizer()

        self._urdf_root_path = urdf_root_path
        
        self._time_step = time_step

        self._old_state = None

        self._obj_type = obj_type

        self._com = pos #centre of mass
        
        self._radius = scale * 0.3 #radius from urdf file

        self.reset(pos, ori, scale, use_fixed_Base)


    def simple_step(self):

        pb.stepSimulation(physicsClientId=self._cid)
        

    def reset(self, pos, ori, scale, use_fixed_Base):

        if self._obj_type == 'cube':

            urdf_file = join(self._urdf_root_path, "cuboid.urdf")

        elif self._obj_type == 'cyl':

             urdf_file = join(self._urdf_root_path, "cylinder.urdf")

        elif self._obj_type == 'cuboid_on_pin':

             urdf_file = join(self._urdf_root_path, "cuboid_on_pin.urdf")

        elif self._obj_type == 'sphere':

             urdf_file = join(self._urdf_root_path, "sphere.urdf") 

        else:

            raise ValueError("Unknown type") 

        self._obj_id   = pb.loadURDF(urdf_file, useFixedBase=use_fixed_Base, globalScaling = scale, physicsClientId=self._cid)

        self.set_base_pose(pos, ori)

        self._num_joints = pb.getNumJoints(self._obj_id, physicsClientId=self._cid)

        if self._num_joints == 0:

            self._sense_jnt_idx = 0

        else:

            self._sense_jnt_idx = range(self._num_joints)[-1]
        
        #disable the default position_control mode. 
        for jointIndex in range (self._num_joints):
            
            pb.setJointMotorControl2(self._obj_id, jointIndex, pb.VELOCITY_CONTROL, targetPosition=0., force=0., physicsClientId=self._cid)

        pb.setRealTimeSimulation(0, physicsClientId=self._cid)

        self.enable_force_torque_sensors()

        self.get_curr_state()

    def set_friction_properties(self, lf=1., sf=1., rf=1., r=0.7):

        pb.changeDynamics(self._obj_id, 0, lateralFriction=lf, spinningFriction=sf, rollingFriction=rf, restitution=r, physicsClientId=self._cid)


    def set_base_pose(self, pos, ori):

        pb.resetBasePositionAndOrientation(self._obj_id, pos, ori, physicsClientId=self._cid)


    def get_mass_matrix(self):

        jnt_state = [0 for _ in range(self._num_joints)]

        jnt_state[self._sense_jnt_idx] = self.get_jnt_state()[0]

        return np.eye(6)*pb.calculateMassMatrix(bodyUniqueId=self._obj_id, objPositions=jnt_state, physicsClientId=self._cid)[0][0]


    def get_base_pose(self):

        return pb.getBasePositionAndOrientation(self._obj_id, physicsClientId=self._cid)


    def get_curr_state(self, ori_type='eul', as_tuple=False):

        link_state = pb.getLinkState(self._obj_id, self._sense_jnt_idx , computeLinkVelocity = 1, physicsClientId=self._cid)

        if as_tuple:
            pos = link_state[0]
        else:
            pos = np.asarray(link_state[0])
        
        if ori_type == 'eul':
            ori = pb.getEulerFromQuaternion(link_state[1])
        
        elif ori_type == 'quat':
            ori = link_state[1]

        elif ori_type == 'mat':
            ori = pb.getMatrixFromQuaternion(link_state[1])

        if not as_tuple:

            ori = np.asarray(ori)

            if ori_type == 'mat':

                ori = ori.reshape(3,3)

        
        vel = np.asarray(link_state[6]) 
        omg = np.asarray(link_state[7])

        if self._old_state is None:

            lin_acc = np.zeros_like(vel)

            ang_acc = np.zeros_like(omg)

        else:

            lin_acc = (vel - self._old_state[2])/self._time_step
            
            ang_acc = (omg - self._old_state[3])/self._time_step


        if as_tuple:

            vel = tuple(vel)
            omg = tuple(omg)
            lin_acc = tuple(lin_acc)
            ang_acc = tuple(ang_acc)

        self._old_state = (pos, ori, vel, omg, lin_acc, ang_acc)

        return  pos, ori, vel, omg, lin_acc, ang_acc


    def enable_force_torque_sensors(self):

        pb.enableJointForceTorqueSensor(self._obj_id, self._sense_jnt_idx , 1, physicsClientId=self._cid)


    def get_jnt_state(self):

        jnt_state = pb.getJointState(self._obj_id, self._sense_jnt_idx, physicsClientId=self._cid)

        #jnt_state[3] = jnt_applied_torque
        #jnt_state[2] = jnt_reaction_forces
        #jnt_state[1] = jnt_vel
        #jnt_state[0] = jnt_pos

        return jnt_state[0], jnt_state[1], jnt_state[2], jnt_state[3]
