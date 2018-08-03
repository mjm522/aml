import os
import sys
import copy
import math
import quaternion
import numpy as np
import pybullet as pb
from aml_rl_envs.aml_rl_robot import AMLRlRobot
from aml_robot.sawyer_kinematics import sawyer_kinematics

class Sawyer(AMLRlRobot):

    def __init__(self, config, cid, jnt_pos = None):

        self._config = config

        self._gripper_index = 7

        self.name = 'right'

        AMLRlRobot.__init__(self, config, cid)

        self.reset(jnt_pos = jnt_pos)

        self._joint_tags = dict(zip(self.get_joint_names(), range(pb.getNumJoints(self._robot_id, physicsClientId=self._cid))))

        self._ee_index = self._joint_tags['right_poking_stick_tip_fixed'] #self._joint_tags['right_j6']

        self._ft_sensor_jnt = self._ee_index#self._joint_tags['right_j6']

        self.enable_force_torque_sensors(joint_idx=self._ft_sensor_jnt)

        self._head_jnt =  1 #self._joint_tags['head_pan']

        self._nu = 7 

    def n_cmd(self):

        return self._nu
 
    def reset(self, base_pos=[0.,0.,0], base_ori=[0.,0.,0.,1.], jnt_pos = None):

        sawyer_path = os.path.join(self._config['urdf_root_path'],"sawyer/sawyer2_with_peg.urdf")

        self._robot_id = pb.loadURDF(sawyer_path, useFixedBase=True, physicsClientId=self._cid)

        # base_pos = [-0.09962212, -0.00962662,  0.03889763]
        # base_ori = [0.004870840187402516, -0.0024265025745719773, -0.7907031013396487, 0.6121756222955841] 

        # pb.resetBasePositionAndOrientation(self._robot_id,[0., 0., 0.0],[0.0,0.0,0.0,1.0], physicsClientId=self._cid)
        pb.resetBasePositionAndOrientation(self._robot_id, base_pos, base_ori, physicsClientId=self._cid)
        
        #tuck position
        # self._jnt_postns=[-5.26523437e-02, -1.18152539e+00, -2.35156250e-03,  2.05699707e+00, 3.74414063e-03,  6.67680664e-01,  3.31310840e+00]
        
        if jnt_pos is None:
            #the position up in the air
            # self._jnt_postns = [0.15703139, -1.64921782,  0.3845158,   1.98039767,  0.01590169,  1.21045897, 2.33553496]
            # self._jnt_postns =  [0.15703139, -1.64921782, 0.3845158, 1.98039767, -0.47909830999996356, 1.21045897, 2.33553496]
            # self._jnt_postns =  [0.15703139, -1.64921782, 0.3845158, 1.98039767, 0.47909830999996356, 1.21045897, 2.33553496]
            #on the table 
            self._jnt_postns = [0.07676213, -1.1887008, 0.29931657, 2.07949845, -0.35975488, 0.66543541, 2.82743442]
        else:
            self._jnt_postns = jnt_pos


        self._movable_jnts = self.get_movable_joints()
        
        self._motor_names = []
        
        self._motor_indices = []
        
        for i in self._movable_jnts:
            
            jnt_info = pb.getJointInfo(self._robot_id, i, physicsClientId=self._cid)
            
            qIndex = jnt_info[3]
            
            if qIndex > -1:

                self._motor_names.append(str(jnt_info[1]))
                
                self._motor_indices.append(i)

        self._kinematics = sawyer_kinematics(self, description=os.path.join(self._config['urdf_root_path'],"sawyer/sawyer.urdf"))

        self.set_ctrl_mode(jnt_postns=self._jnt_postns)

    def state(self, ori_type = 'quat'):

        ee_pos, ee_ori = self.get_ee_pose()

        ee_vel, ee_omg = self.get_ee_velocity()

        jnt_poss, jnt_vels, jnt_reaction_forces, jnt_applied_torques = self.get_jnt_state()

        jac = self.jacobian(jnt_poss)
        M, Mee_inv = self.get_ee_mass_matrix(jac=jac, jnt_pos=jnt_poss)

        state = {}
        state['position'] = np.asarray(jnt_poss)
        state['velocity'] = np.asarray(jnt_vels)
        state['effort'] = np.asarray(jnt_reaction_forces)
        state['jacobian'] = jac
        state['inertia'] = M
        state['Mee_inv'] = Mee_inv
        state['ee_point'] = ee_pos
        
        if ori_type != 'quat':
            state['ee_ori'] = np.asarray(pb.getEulerFromQuaternion(ee_ori))
        else:
            state['ee_ori'] = np.array([ee_ori[3],ee_ori[0],ee_ori[1],ee_ori[2]])

        state['ee_vel'] = ee_vel
        state['ee_omg'] = ee_omg

        # state['gripper_state'] = self.gripper_state()

        return state


    def get_action_dim(self):

        return 9 #position x,y,z and roll/pitch/yaw euler angles of end effector

    def get_obs_dim(self):

        return len(self.get_observation())

    def get_observation(self):
        
        observation = []
        
        jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque = self.get_jnt_state()
        
        pos, ori, vel, omg  = self.ee_state()
        
        euler = pb.getEulerFromQuaternion(ori)
        
        observation.extend(jnt_pos)
        
        observation.extend(jnt_vel)

        observation.extend(list(pos))
        # observation.extend(list(euler))
        observation.extend(list(vel))
        # observation.extend(list(ee_vel[1]))

        return observation

    def ee_state(self, as_tuple=True):

        return self.get_ee_state(self._ee_index, as_tuple)

    def get_ee_wrench(self, local=False):
        '''
            End effector forces and torques.
            Returns [fx, fy, fz, tx, ty, tz]
        '''

        _, _,jnt_reaction_force, _ = self.get_jnt_state(self._ft_sensor_jnt)

        if local:
            ee_pos, ee_ori, _, _ = self.get_ee_state(self._ft_sensor_jnt)
            jnt_reaction_force = np.asarray(jnt_reaction_force)
            force  = tuple(jnt_reaction_force[:3])
            torque = tuple(jnt_reaction_force[3:])

            inv_ee_pos, inv_ee_ori = pb.invertTransform(ee_pos, ee_ori)
            
            force, _  = pb.multiplyTransforms(inv_ee_pos, inv_ee_ori, force, (0,0,0,1))
            torque, _ = pb.multiplyTransforms(inv_ee_pos, inv_ee_ori, torque, (0,0,0,1))
            jnt_reaction_force = force + torque

        return jnt_reaction_force

    def inv_kin(self, ee_pos, ee_ori=None):
        #this is assuming that the head pan joint has been made "fixed"
        cmd = self.get_ik(ee_idx=self._ee_index, ee_pos=ee_pos, ee_ori=ee_ori)
        #we delete the corresponding head ik from this
        return np.asarray(cmd) #np.delete(cmd, 0, 0)


    def get_ee_mass_matrix(self, jac, jnt_pos):
        #this is on the assumption that
        #the head joint is not included

        joint_pos = np.zeros(21)

        joint_pos[:7] = jnt_pos

        M =  self.get_mass_matrix(tuple(joint_pos))

        #removing the head axis dependency
        M = M.reshape(7,7)
        # M = np.delete(M, self._head_jnt, axis=0)
        # M = np.delete(M, self._head_jnt, axis=1)

        return M, np.dot( np.dot( jac, np.linalg.pinv(M) ), jac.T )


    def set_joint_state(self, joint_state):
        
        for k in range(len(self._movable_jnts)):
            
            pb.resetJointState(self._robot_id, self._movable_jnts[k], joint_state[k], physicsClientId=self._cid)

    def get_ee_pose(self, as_tuple=False):

        link_state = pb.getLinkState(self._robot_id, self._ee_index, physicsClientId=self._cid)

        ee_pos = link_state[0]
        ee_ori = link_state[1]

        if not as_tuple:
            ee_pos = np.asarray(ee_pos) 
            ee_ori = np.asarray(ee_ori)

        return ee_pos, ee_ori

    def get_ee_velocity(self):

        link_state = pb.getLinkState(self._robot_id, self._ee_index, computeLinkVelocity = 1, physicsClientId=self._cid)

        ee_vel = np.asarray(link_state[6]) 
        
        ee_omg = np.asarray(link_state[7])

        return ee_vel, ee_omg 

    # def get_jnt_state(self):

    #     num_jnts = len(self._motor_indices)

    #     jnt_pos = []
        
    #     jnt_vel = []
        
    #     jnt_reaction_forces =  []
        
    #     jnt_applied_torque  = []

    #     for jnt_idx in range(len(self._motor_indices)):

    #         jnt_state = pb.getJointState(self._robot_id, self._motor_indices[jnt_idx], physicsClientId=self._cid)
            
    #         jnt_pos.append(jnt_state[0])
            
    #         jnt_vel.append(jnt_state[1])
            
    #         jnt_reaction_forces.append(jnt_state[2])
            
    #         jnt_applied_torque.append(jnt_state[3])

    #     return jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque

    def apply_action(self, motor_commands, Kp=None, Kd=None):

        for action in range (len(motor_commands)):

            motor = self._motor_indices[action]

            if Kp is None:
                self.apply_ctrl(motor, motor_commands[action])
            else:
                if Kd is None:
                    self.apply_ctrl(motor, motor_commands[action], Kp[action])
                else:
                    self.apply_ctrl(motor, motor_commands[action], Kp[action], Kd[action])


    def joint_names(self):

        return ['right_j%s' % (s,) for s in range(0, 7)]

    def jacobian(self, joint_angles=None):

        if len(joint_angles) < 21:

            joint_pos = np.zeros(21)

            joint_pos[:7] = joint_angles
        else:
            joint_pos=  joint_angles


        linear_jacobian, angular_jacobian = pb.calculateJacobian(bodyUniqueId=self._robot_id,
                                                                 linkIndex=self._ee_index,
                                                                 localPosition=[0.,0.,0.],
                                                                 objPositions=list(joint_pos),
                                                                 objVelocities=np.zeros(len(joint_pos)).tolist(),
                                                                 objAccelerations=np.zeros(len(joint_pos)).tolist(),
                                                                 physicsClientId=self._cid)
    

        # jac = np.hstack([np.asarray(linear_jacobian).reshape(3,7), np.asarray(langular_jacobian).reshape(3,7)])

        lin_jac = np.asarray(linear_jacobian).reshape(3,7)

        ang_jac = np.asarray(angular_jacobian).reshape(3,7)

        #
        # return np.delete(jacobian, 1, 1)

        # if joint_angles is None:

        #     argument = dict(zip(self.joint_names(), self.get_jnt_state()[0]))

        # else:

        #     argument = dict(zip(self.joint_names(), joint_angles))
        # # combine the names and joint angles to a dictionary, that only is accepted by kdl
        # jacobian = np.array(self._kinematics.jacobian(argument))

        # print jacobian

        return np.vstack([lin_jac, ang_jac])
