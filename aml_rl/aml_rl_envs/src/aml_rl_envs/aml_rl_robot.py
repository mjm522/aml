import numpy as np
import pybullet as pb


class AMLRlRobot(object):

    def __init__(self, config, robot_id):

        self._config = config

        self._urdf_root_path = self._config['urdf_root_path']

        self._time_step = self._config['time_step']
        
        self._max_velocity = self._config['max_velocity']
        
        self._max_force = self._config['max_force']
        
        self._ctrl_type = self._config['ctrl_type']

        self._robot_id = robot_id

    def simple_step(self):
        
        pb.stepSimulation()

    def set_base_pose(self, pos, ori):

        pb.resetBasePositionAndOrientation(self._robot_id, pos, ori)
        

    def set_ctrl_mode(self):

        self._tot_num_jnts = pb.getNumJoints(self._robot_id)

        self._jnt_indexs = self.get_movable_joints()

        self._jnt_postns = [0. for _ in range(len(self._jnt_indexs))]
        
        #disable the default position_control mode. 
        for k, jnt_index in enumerate(self._jnt_indexs):
            
            pb.resetJointState(self._robot_id, jnt_index, self._jnt_postns[k])

            if self._ctrl_type == 'pos':
                
                pb.setJointMotorControl2(self._robot_id, jnt_index, pb.POSITION_CONTROL, targetPosition=self._jnt_postns[k], force=self._max_force)
            
            else:
                
                pb.setJointMotorControl2(self._robot_id, jnt_index, pb.VELOCITY_CONTROL, targetPosition=self._jnt_postns[k], force=0.5)

    
    def set_friction_properties(self, lf=1., sf=1., rf=1., r=0.7):

        for jnt_idx in self._jnt_indexs:

            pb.changeDynamics(self._robot_id, jnt_idx,   lateralFriction=lf, spinningFriction=sf, rollingFriction=rf, restitution=r)


    def enable_force_torque_sensors(self, joint_idx = -2):

        for i in self._jnt_indexs:

            pb.enableJointForceTorqueSensor(self._robot_id, i, 1)


    def get_movable_joints(self):

        movable_jnts = []
        
        for i in range (self._tot_num_jnts):
            
            jnt_info = pb.getJointInfo(self._robot_id, i)
            
            qIndex = jnt_info[3]
            
            if qIndex > -1 and jnt_info[1] != "head_pan":
                
                movable_jnts.append(i)

        return movable_jnts


    def apply_ctrl(self, motor, cmd, Kp=None):

        if self._ctrl_type == 'torque':
                
                pb.setJointMotorControl2(self._robot_id, motor, pb.TORQUE_CONTROL, force=cmd)
            
        elif self._ctrl_type == 'pos':
            
            if Kp is None:
                
                pb.setJointMotorControl2(self._robot_id, motor, pb.POSITION_CONTROL, targetPosition=cmd, force=self._max_force)
            
            else:
                
                pb.setJointMotorControl2(self._robot_id, motor, pb.POSITION_CONTROL, targetPosition=cmd, positionGain=Kp[motor], force=self._max_force)
        
        elif self._ctrl_type == 'vel':
            
            pb.setJointMotorControl2(self._robot_id, motor, pb.VELOCITY_CONTROL, targetVelocity=cmd, force=self._max_force)


    def get_ik(self, ee_idx, ee_pos, ee_ori=None):

        if ee_ori is None:

            return pb.calculateInverseKinematics(self._robot_id, 
                                                ee_idx, 
                                                targetPosition=ee_pos)
        
        else:
            
            return pb.calculateInverseKinematics(self._robot_id, 
                                                ee_idx, 
                                                targetPosition=ee_pos, 
                                                targetOrientation=ee_ori)


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
                                          objAccelerations=js_acc)
        
        return np.asarray(tau)

