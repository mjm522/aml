import os
import copy
import math
import numpy as np
import pybullet as pb
from os.path import exists, join
from aml_rl_envs.config import urdf_root_path

class ManObject():

    def __init__(self, urdf_root_path=urdf_root_path, time_step=0.01, pos = [1.75, 0., 1.5], ori = [0., 0., 0., 1], j_pos=[0.0, 0.0], scale=1., useFixedBase=True):

        self._urdf_root_path = urdf_root_path
        
        self._time_step = time_step

        self._old_state = None

        self._com = pos #centre of mass
        
        self._radius = scale * 0.3 #radius from urdf file

        self.reset(pos, ori, scale, useFixedBase)
        

    def reset(self, pos, ori, scale, useFixedBase):

        self._obj_id   = pb.loadURDF(join(self._urdf_root_path, "object.urdf"), useFixedBase=useFixedBase, globalScaling = scale)

        pb.resetBasePositionAndOrientation(self._obj_id, pos, ori)

        self._num_joints = pb.getNumJoints(self._obj_id)
        
        #disable the default position_control mode. 
        for jointIndex in range (self._num_joints):
            
            pb.setJointMotorControl2(self._obj_id, jointIndex, pb.VELOCITY_CONTROL, targetPosition=0., force=0.)

        pb.setRealTimeSimulation(0)

        self.enable_force_torque_sensors()

        self.get_curr_state()

    def set_friction_properties(self, lf=1., sf=1., rf=1., r=0.7):

        pb.changeDynamics(self._obj_id, 0, lateralFriction=lf, spinningFriction=sf, rollingFriction=rf, restitution=r)


    def get_mass_matrix(self):

        jnt_state = [0., self.get_jnt_state()[0]]

        return np.eye(6)*pb.calculateMassMatrix(bodyUniqueId=self._obj_id, objPositions=jnt_state)[0][0]


    def get_base_pose(self):

        return pb.getBasePositionAndOrientation(self._obj_id)


    def get_curr_state(self, ori_as_euler=True):

        link_state = pb.getLinkState(self._obj_id, 1, computeLinkVelocity = 1)

        pos = np.asarray(link_state[0])
        
        if ori_as_euler:
            
            ori = np.asarray(pb.getEulerFromQuaternion(link_state[1]))
        
        else:
            
            ori = np.asarray(pb.getMatrixFromQuaternion(link_state[1])).reshape(3,3)
        
        vel = np.asarray(link_state[6]) 
        
        omg = np.asarray(link_state[7])

        if self._old_state is None:
            
            lin_acc = np.zeros_like(vel)
            
            ang_acc = np.zeros_like(omg)
        
        else:
            
            lin_acc = (vel - self._old_state[2])/self._time_step
            
            ang_acc = (omg - self._old_state[3])/self._time_step

        self._old_state = (pos, ori, vel, omg, lin_acc, ang_acc)

        return  pos, ori, vel, omg, lin_acc, ang_acc


    def enable_force_torque_sensors(self):

        pb.enableJointForceTorqueSensor(self._obj_id, 1, 1)


    def get_jnt_state(self):

        jnt_state = pb.getJointState(self._obj_id, 1)

        #jnt_state[3] = jnt_applied_torque
        #jnt_state[2] = jnt_reaction_forces
        #jnt_state[1] = jnt_vel
        #jnt_state[0] = jnt_pos

        return jnt_state[0], jnt_state[1], jnt_state[2], jnt_state[3]
