import os
import copy
import math
import numpy as np
import pybullet as pb
from aml_io.log_utils import aml_logging


class ManipulableObject(object):

    def __init__(self, urdf_path=None, time_step=0.01,
                       pos = [1.75, 0., 1.5], ori = [0., 0., 0., 1], 
                       j_pos=[0.0, 0.0], scale=1., use_fixed_base=True, render=False):

        self._urdf_path = urdf_path
        
        self._time_step = time_step

        self._old_state = None

        self._com = pos #centre of mass
        
        self._scale = scale #radius from urdf file

        self._use_fixed_base = use_fixed_base

        self.load()

        self.reset(pos, ori)

        aml_logging.setup('all')


    def load(self):

        self._obj_id   = pb.loadURDF(self._urdf_path, useFixedBase=self._use_fixed_base, globalScaling = self._scale)
        self._num_joints = pb.getNumJoints(self._obj_id)

        # self._sense_jnt_idx = range(self._num_joints)[-1]
        aml_logging.setup('all')
        aml_logging.info("Object id: %d"%(self._obj_id,))
        aml_logging.info("Object num joints: %d"%(self._num_joints,))

        #         #disable the default position_control mode. 
        # for joint_index in range(self._num_joints):
        #     pb.setJointMotorControl2(self._obj_id, joint_index, pb.VELOCITY_CONTROL, targetPosition=0., force=0.)


        # self.enable_force_torque_sensors()

    def simple_step(self):

        pb.stepSimulation()
        

    def reset(self, pos, ori):
        # doing nothing at the moment
        self.set_base_pose(pos, ori)

    def set_friction_properties(self, lf=1., sf=1., rf=1., r=0.7):

        pb.changeDynamics(self._obj_id, 0, lateralFriction=lf, spinningFriction=sf, rollingFriction=rf, restitution=r)


    def set_base_pose(self, pos, ori):

        pb.resetBasePositionAndOrientation(self._obj_id, pos, ori)


    def get_mass_matrix(self):

        jnt_state = [0 for _ in range(self._num_joints)]

        jnt_state[self._sense_jnt_idx] = self.get_jnt_state()[0]

        return np.eye(6)*pb.calculateMassMatrix(bodyUniqueId=self._obj_id, objPositions=jnt_state)[0][0]


    def get_base_pose(self):

        return pb.getBasePositionAndOrientation(self._obj_id)


    def get_curr_state(self, ori_type='eul'):

        link_state = pb.getLinkState(self._obj_id, self._sense_jnt_idx , computeLinkVelocity = 1)

        pos = np.asarray(link_state[0])
        
        if ori_type == 'eul':
            ori = np.asarray(pb.getEulerFromQuaternion(link_state[1]))
        
        elif ori_type == 'quat':
            ori = np.asarray(link_state[1])

        elif ori_type == 'mat':
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


    # def enable_force_torque_sensors(self):

    #     pb.enableJointForceTorqueSensor(self._obj_id, self._sense_jnt_idx , 1)


    def get_jnt_state(self):

        jnt_state = pb.getJointState(self._obj_id, self._sense_jnt_idx )

        #jnt_state[3] = jnt_applied_torque
        #jnt_state[2] = jnt_reaction_forces
        #jnt_state[1] = jnt_vel
        #jnt_state[0] = jnt_pos

        return jnt_state[0], jnt_state[1], jnt_state[2], jnt_state[3]
