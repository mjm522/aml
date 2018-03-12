import os
import copy
import math
import numpy as np
import pybullet as pb
from aml_rl_envs.aml_rl_robot import AMLRlRobot

#Ref:https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/bullet

class Kuka(AMLRlRobot):

    def __init__(self, config):
        
        self._fingerA_force = 2 
        
        self._fingerB_force = 2.5
        
        self._finger_tip_force = 2
        
        self._use_inverse_kinematics = 0
        
        self._use_simulation = True
        
        self._use_null_space = True
        
        self._use_orientation = True
        
        self._ee_index = 13
        
        self._gripper_index = 7

        AMLRlRobot.__init__(self, config)

        #lower limits for null space
        self._ll=[-.967,-2 ,-2.96,0.19,-2.96,-2.09,-3.05]
        #upper limits for null space
        self._ul=[.967,2 ,2.96,2.29,2.96,2.09,3.05]
        #joint ranges for null space
        self._jr=[5.8,4,5.8,4,5.8,4,6]
        #restposes for null space
        self._rp=[0,0,0,0.5*math.pi,0,-math.pi*0.5*0.66,0]
        #joint damping coefficents
        self._jd=[0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001]
        
        self.reset()
        
    def reset(self):
        
        objects = pb.loadSDF(os.path.join(self._urdf_root_path,"kuka/kuka_with_gripper2.sdf"))
        
        self._robot_id = objects[0]

        pb.resetBasePositionAndOrientation(self._robot_id,
                                           [-0.100000,0.000000,0.070000],
                                           [0.000000,0.000000,0.000000,1.000000])
        
        self._jnt_postns = [ 0.006418, 0.413184, -0.011401, 
                                -1.589317, 0.005379, 1.137684,
                                -0.006539, 0.000048, -0.299912, 
                                 0.000000, -0.000043, 0.299960, 
                                 0.000000, -0.000200 ]
        
        
        
        self._tray_id = pb.loadURDF(os.path.join(self._urdf_root_path,"tray/tray.urdf"), 
                                    0.640000,0.075000, 0.050000,0.000000,0.000000,1.000000,0.000000)
        
        self._ee_pos = [0.537,0.0,0.5]
        
        self._ee_ori = 0

        self._jnt_indexs = self.get_movable_joints()

        self._num_jnts = len(self._jnt_indexs)
        
        self._motor_names = []
        
        self._motor_indices = []
        
        for i in range (self._num_jnts):
            
            joint_info = pb.getJointInfo(self._robot_id,i)
            
            q_index = joint_info[3]
            
            if q_index > -1:

                self._motor_names.append(str(joint_info[1]))
                
                self._motor_indices.append(i)

        self.set_ctrl_mode()

    def get_action_dim(self):
        
        if (self._use_inverse_kinematics):
            
            return len(self._motor_indices)
        
        return 6 

    def get_obs_dim(self):
        
        return len(self.get_observation())

    def get_observation(self):
        
        observation = []

        state = pb.getLinkState(self._robot_id, self._gripper_index)

        pos = state[0]

        orn = state[1]

        euler = pb.getEulerFromQuaternion(orn)
                
        observation.extend(list(pos))

        observation.extend(list(euler))
        
        return observation

    def apply_action(self, motor_commands):
        
        #print ("self._num_jnts")
        #print (self._num_jnts)
        if (self._use_inverse_kinematics):
            
            dx = motor_commands[0]
            dy = motor_commands[1]
            dz = motor_commands[2]
            da = motor_commands[3]

            fingerAngle = motor_commands[4]
            
            state = pb.getLinkState(self._robot_id,self._ee_index)
            
            actual_ee_pos = state[0]
     
            self._ee_pos[0] = self._ee_pos[0]+dx
            
            if (self._ee_pos[0]>0.65):
                
                self._ee_pos[0]=0.65
            
            if (self._ee_pos[0]<0.50):
                
                self._ee_pos[0]=0.50
            
            self._ee_pos[1] = self._ee_pos[1]+dy
            
            if (self._ee_pos[1]<-0.17):
                
                self._ee_pos[1]=-0.17
            
            if (self._ee_pos[1]>0.22):
                
                self._ee_pos[1]=0.22
            
            self._ee_pos[2] = self._ee_pos[2]+dz
        
            self._ee_ori = self._ee_ori + da
            
            pos = self._ee_pos
            
            orn = pb.getQuaternionFromEuler([0,-math.pi,0]) 
            
            if self._use_null_space:
                
                if self._use_orientation:
                    
                    jnt_poses = pb.calculateInverseKinematics(self._robot_id,self._ee_index,pos,orn,self._ll,self._ul,self._jr,self._rp)
                
                else:
                    
                    jnt_poses = pb.calculateInverseKinematics(self._robot_id,self._ee_index,pos,lowerLimits=self._ll, upperLimits=self._ul, jointRanges=self._jr, restPoses=self._rp)
            else:
                
                if self._use_orientation:
                    
                    jnt_poses = pb.calculateInverseKinematics(self._robot_id,self._ee_index,pos,orn,jointDamping=self._jd)
                
                else:
                    
                    jnt_poses = pb.calculateInverseKinematics(self._robot_id,self._ee_index,pos)
        

            if (self._use_simulation):
                
                for i in range (self._ee_index+1):
       
                    pb.setJointMotorControl2(bodyUniqueId=self._robot_id, 
                                             jointIndex=i, 
                                             controlMode=pb.POSITION_CONTROL, 
                                             targetPosition=jnt_poses[i],
                                             targetVelocity=0, 
                                             force=self._max_force, 
                                             maxVelocity=self._max_velocity, 
                                             positionGain=0.3,velocityGain=1)
            else:
                #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
                for i in range (self._num_jnts):
                    
                    pb.resetJointState(self._robot_id,i,jnt_poses[i])
            
            #fingers
            pb.setJointMotorControl2(self._robot_id,7,pb.POSITION_CONTROL, targetPosition=self._ee_ori,force=self._max_force)
            
            pb.setJointMotorControl2(self._robot_id,8,pb.POSITION_CONTROL, targetPosition=-fingerAngle,force=self._fingerA_force)
            
            pb.setJointMotorControl2(self._robot_id,11,pb.POSITION_CONTROL, targetPosition=fingerAngle,force=self._fingerB_force)
            
            pb.setJointMotorControl2(self._robot_id,10,pb.POSITION_CONTROL, targetPosition=0,force=self._finger_tip_force)
            
            pb.setJointMotorControl2(self._robot_id,13,pb.POSITION_CONTROL, targetPosition=0,force=self._finger_tip_force)
            
            
        else:
            
            for action in range (len(motor_commands)):
                
                motor = self._motor_indices[action]
                
                pb.setJointMotorControl2(self._robot_id,motor,pb.POSITION_CONTROL,targetPosition=motor_commands[action],force=self._max_force)