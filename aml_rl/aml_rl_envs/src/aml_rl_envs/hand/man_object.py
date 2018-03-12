import os
import copy
import math
import numpy as np
import pybullet as pb
from os.path import exists, join
from aml_rl_envs.aml_rl_env import AMLRlEnv
from aml_rl_envs.config import AML_RL_ENV_CONFIG


class ManObject(AMLRlEnv):

    def __init__(self, pos = [1.75, 0., 1.5], ori = [0., 0., 0., 1], 
                       j_pos=[0.0, 0.0], scale=1., use_fixed_Base=True, 
                       config=AML_RL_ENV_CONFIG, obj_type='cyl'):

        self._config = config

        AMLRlEnv.__init__(self, config, set_gravity=False)

        self._urdf_root_path = config['urdf_root_path']
        
        self._time_step = config['time_step']

        self._old_state = None

        self._obj_type = 'cyl'

        self._com = pos #centre of mass
        
        self._radius = scale * 0.3 #radius from urdf file

        self.reset(pos, ori, scale, use_fixed_Base)
        

    def reset(self, pos, ori, scale, use_fixed_Base):

        if self._obj_type == 'cube':

            urdf_file = join(self._urdf_root_path, "cylinder.urdf")

        elif self._obj_type == 'cyl':

             urdf_file = join(self._urdf_root_path, "cuboid.urdf") 

        else:

            raise ValueError("Unknown type") 

        self._obj_id   = pb.loadURDF(urdf_file, useFixedBase=use_fixed_Base, globalScaling = scale)

        self.set_base_pose(pos, ori)

        self._num_joints = pb.getNumJoints(self._obj_id)
        
        #disable the default position_control mode. 
        for jointIndex in range (self._num_joints):
            
            pb.setJointMotorControl2(self._obj_id, jointIndex, pb.VELOCITY_CONTROL, targetPosition=0., force=0.)

        pb.setRealTimeSimulation(0)

        self.enable_force_torque_sensors()

        self.get_curr_state()

    def set_friction_properties(self, lf=1., sf=1., rf=1., r=0.7):

        pb.changeDynamics(self._obj_id, 0, lateralFriction=lf, spinningFriction=sf, rollingFriction=rf, restitution=r)


    def set_base_pose(self, pos, ori):

        pb.resetBasePositionAndOrientation(self._obj_id, pos, ori)


    def get_mass_matrix(self):

        jnt_state = [0., self.get_jnt_state()[0]]

        return np.eye(6)*pb.calculateMassMatrix(bodyUniqueId=self._obj_id, objPositions=jnt_state)[0][0]


    def get_base_pose(self):

        return pb.getBasePositionAndOrientation(self._obj_id)


    def get_curr_state(self, ori_as_euler=True):

        link_state = pb.getLinkState(self._obj_id, range(self._num_joints)[-1], computeLinkVelocity = 1)

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

        pb.enableJointForceTorqueSensor(self._obj_id, range(self._num_joints)[-1], 1)


    def get_jnt_state(self):

        jnt_state = pb.getJointState(self._obj_id, 1)

        #jnt_state[3] = jnt_applied_torque
        #jnt_state[2] = jnt_reaction_forces
        #jnt_state[1] = jnt_vel
        #jnt_state[0] = jnt_pos

        return jnt_state[0], jnt_state[1], jnt_state[2], jnt_state[3]
