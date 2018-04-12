import numpy as np
import pybullet as pb
from aml_rl_envs.bullet_visualizer import setup_bullet_visualizer

class AMLRlRobot(object):

    def __init__(self, config, cid, robot_id=None):

        self._config = config

        self._urdf_root_path = self._config['urdf_root_path']

        self._time_step = self._config['time_step']
        
        self._max_velocity = self._config['max_velocity']
        
        self._max_force = self._config['max_force']
        
        self._ctrl_type = self._config['ctrl_type']

        self._cid = cid

        if 'call_renderer' in self._config.keys():

            if self._config['call_renderer']:
                
                self._cid = setup_bullet_visualizer(self._renders)


        self._robot_id = robot_id

    def simple_step(self):
        
        pb.stepSimulation(physicsClientId=self._cid)

    def set_base_pose(self, pos, ori):

        pb.resetBasePositionAndOrientation(self._robot_id, pos, ori, physicsClientId=self._cid)

    def get_ee_state(self, ee_idx, as_tuple=True):
        """
        returns orientation in bullet format quaternion [x,y,z,w]
        """

        link_state = pb.getLinkState(self._robot_id, ee_idx, computeLinkVelocity = 1, physicsClientId=self._cid)

        if not as_tuple:
            ee_pos = np.asarray(link_state[0])
            ee_ori = np.asarray(link_state[1])
            ee_vel = np.asarray(link_state[2])
            ee_omg = np.asarray(link_state[3])
        else:
            ee_pos = link_state[0]
            ee_ori = link_state[1]
            ee_vel = link_state[2]
            ee_omg = link_state[3]

        return ee_pos, ee_ori, ee_vel, ee_omg
        
    def set_ctrl_mode(self, jnt_postns=None):

        self._tot_num_jnts = pb.getNumJoints(self._robot_id, physicsClientId=self._cid)

        self._jnt_indexs = self.get_movable_joints()

        if jnt_postns is None:

            self._jnt_postns = self.get_jnt_state()[0]
        
        else:
            assert(len(self._jnt_indexs) == len(jnt_postns))
            
            self._jnt_postns = jnt_postns

        #disable the default position_control mode. 
        for k, jnt_index in enumerate(self._jnt_indexs):
            
            pb.resetJointState(self._robot_id, jnt_index, self._jnt_postns[k], physicsClientId=self._cid)

            if self._ctrl_type == 'pos':
                
                pb.setJointMotorControl2(self._robot_id, jnt_index, pb.POSITION_CONTROL, targetPosition=self._jnt_postns[k], force=self._max_force, physicsClientId=self._cid)
            
            else:
                
                pb.setJointMotorControl2(self._robot_id, jnt_index, pb.VELOCITY_CONTROL, targetPosition=self._jnt_postns[k], force=0.5, physicsClientId=self._cid)

    
    def set_friction_properties(self, lf=1., sf=1., rf=1., r=0.7):

        for jnt_idx in self._jnt_indexs:

            pb.changeDynamics(self._robot_id, jnt_idx,   lateralFriction=lf, spinningFriction=sf, rollingFriction=rf, restitution=r, physicsClientId=self._cid)


    def enable_force_torque_sensors(self, joint_idx = -2):

        for i in self._jnt_indexs:

            pb.enableJointForceTorqueSensor(self._robot_id, i, 1, physicsClientId=self._cid)


    def get_movable_joints(self):

        movable_jnts = []
        
        for i in range (pb.getNumJoints(self._robot_id, physicsClientId=self._cid)):
            
            jnt_info = pb.getJointInfo(self._robot_id, i, physicsClientId=self._cid)
            
            qIndex = jnt_info[3]
            
            if qIndex > -1 and jnt_info[1] != "head_pan":
                
                movable_jnts.append(i)

        return movable_jnts


    def apply_ctrl(self, motor, cmd, Kp=None):

        if self._ctrl_type == 'torque':

            pb.setJointMotorControl2(self._robot_id, motor, pb.TORQUE_CONTROL, force=cmd, physicsClientId=self._cid)
            
        elif self._ctrl_type == 'pos':
            
            if Kp is None:
                
                pb.setJointMotorControl2(self._robot_id, motor, pb.POSITION_CONTROL, targetPosition=cmd, force=self._max_force, physicsClientId=self._cid)
            
            else:
                
                pb.setJointMotorControl2(self._robot_id, motor, pb.POSITION_CONTROL, targetPosition=cmd, positionGain=Kp[motor], force=self._max_force, physicsClientId=self._cid)
        
        elif self._ctrl_type == 'vel':
            
            pb.setJointMotorControl2(self._robot_id, motor, pb.VELOCITY_CONTROL, targetVelocity=cmd, force=self._max_force, physicsClientId=self._cid)

        else:

            raise Exception("Unknown control type ...")


    def get_ik(self, ee_idx, ee_pos, ee_ori=None):

        if ee_ori is None:

            return pb.calculateInverseKinematics(self._robot_id, 
                                                ee_idx, 
                                                targetPosition=ee_pos,
                                                physicsClientId=self._cid)
        
        else:
            
            return pb.calculateInverseKinematics(self._robot_id, 
                                                ee_idx, 
                                                targetPosition=ee_pos, 
                                                targetOrientation=ee_ori,
                                                physicsClientId=self._cid)


    def get_inv_dyn(self, js_pos, js_vel, js_acc=None):

        if js_acc is None:

            js_acc = [0. for _ in range(len(js_pos))]

        if (not isinstance(js_pos, tuple)) or (not isinstance(js_pos, list)):

            js_pos = tuple(js_pos)

        if (not isinstance(js_vel, tuple)) or (not isinstance(js_vel, list)):
            
            js_vel = tuple(js_vel)

        if (not isinstance(js_acc, tuple)) or (not isinstance(js_acc, list)):
            
            js_acc = tuple(js_acc)

        tau = pb.calculateInverseDynamics(bodyUniqueId=self._robot_id,
                                          objPositions=js_pos,
                                          objVelocities=js_vel,
                                          objAccelerations=js_acc,
                                          physicsClientId=self._cid)
        
        return np.asarray(tau)


    def get_jnt_state(self):

        jnt_poss = []
        jnt_vels = []
        jnt_reaction_forces =  []
        jnt_applied_torques  = []

        for jnt_idx in self._jnt_indexs:

            jnt_state = pb.getJointState(self._robot_id, jnt_idx)
            
            jnt_poss.append(jnt_state[0])
            
            jnt_vels.append(jnt_state[1])
            
            jnt_reaction_forces.append(jnt_state[2])
            
            jnt_applied_torques.append(jnt_state[3])


        return jnt_poss, jnt_vels, jnt_reaction_forces, jnt_applied_torques

    
    def apply_jnt_ctrl(self, cmd, Kp=None):

        assert len(self._jnt_indexs) == len(cmd)

        for j, idx in enumerate(self._jnt_indexs):

            self.apply_ctrl(idx, cmd[j])


    def get_joint_limits(self):

        num_jnts = len(self._jnt_indexs)

        lower_lim = np.zeros(num_jnts)

        upper_lim = np.zeros(num_jnts)

        mean_ = np.zeros(num_jnts)

        range_ = np.zeros(num_jnts)

        for k, idx in enumerate(self._jnt_indexs):

            lower_lim[k] = pb.getJointInfo(self._robot_id, idx, physicsClientId=self._cid)[8]
            
            upper_lim[k] = pb.getJointInfo(self._robot_id, idx, physicsClientId=self._cid)[9]

            mean_[k] = 0.5*( lower_lim[k] + upper_lim[k] )

            range_[k] = ( upper_lim[k] - lower_lim[k])
                
        return {'lower': lower_lim, 'upper':upper_lim, 'mean':mean_, 'range': range_}

