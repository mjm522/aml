import numpy as np
import pybullet as pb
from aml_rl_envs.aml_rl_robot import AMLRlRobot
from aml_rl_envs.config import AML_RL_ROBOT_CONFIG


class AMLRlHand(AMLRlRobot):

    def __init__(self, config, num_fingers, finger_jnt_indices=None, robot_id=None):

        AMLRlRobot.__init__(self, config, robot_id)

        self._config = config

        self._num_fingers = num_fingers

        self._finger_jnt_indices = finger_jnt_indices

    def setup_hand(self):

        self.set_ctrl_mode()

        self._motor_names   = [[] for _ in range(self._num_fingers)]
        
        for k in range(self._num_fingers):

            for jnt_idx in self._finger_jnt_indices[k]:
                
                jnt_info = pb.getJointInfo(self._robot_id, jnt_idx)
                
                q_index = jnt_info[2]

                self._motor_names[k].append(str(jnt_info[1]))
                
        pb.setRealTimeSimulation(0)

    #old name is set_joint_state
    def set_fin_joint_state(self, finger_idx, jnt_pos):

        for k, idx in enumerate(self._finger_jnt_indices[finger_idx]):

            pb.resetJointState(self._robot_id, idx, jnt_pos[k])

    
    def apply_action(self, finger_idx, motor_commands, Kp=None):

        for k in range (len(motor_commands)):

            motor = self._finger_jnt_indices[finger_idx][k]

            self.apply_ctrl(motor, motor_commands[k], Kp)


    def inv_kin(self, finger_idx, ee_pos, ee_ori=None):

        return self.get_ik(self._ee_indexs[finger_idx], 
                            ee_pos, ee_ori)[3*finger_idx:3*finger_idx+3]


    def get_finger_limits(self):

        lower_lim_list = []; upper_lim_list = []; mean_list = []; range_list = []

        for j in range(self._num_fingers):

            num_jnts = len(self._finger_jnt_indices[j])

            lower_lim = np.zeros(num_jnts)

            upper_lim = np.zeros(num_jnts)

            mean_ = np.zeros(num_jnts)

            range_ = np.zeros(num_jnts)

            for k, idx in enumerate(self._finger_jnt_indices[j]):

                lower_lim[k] = pb.getJointInfo(self._robot_id, idx)[8]
                
                upper_lim[k] = pb.getJointInfo(self._robot_id, idx)[9]

                mean_[k] = 0.5*( lower_lim[k] + upper_lim[k] )

                range_[k] = ( upper_lim[k] - lower_lim[k])

            lower_lim_list.append(lower_lim)

            upper_lim_list.append(upper_lim)

            mean_list.append(mean_)

            range_list.append(range_)
                
        return {'lower': lower_lim_list, 'upper':upper_lim_list, 'mean':mean_list, 'range': range_list}


    def get_ee_states(self, as_tuple=False):

        ee_poss = []
        ee_oris = []
        ee_vels = []
        ee_omgs = []

        for k in range(self._num_fingers):

            link_state = pb.getLinkState(self._robot_id, self._ee_indexs[k], computeLinkVelocity = 1)

            if as_tuple:
                
                ee_poss.append(link_state[0])
                ee_oris.append(link_state[1])
                ee_vels.append(link_state[6])
                ee_omgs.append(link_state[7])
            
            else:
                
                ee_poss.append(np.asarray(link_state[0]))
                ee_oris.append(np.asarray(link_state[1]))
                ee_vels.append(np.asarray(link_state[6]))
                ee_omgs.append(np.asarray(link_state[7]))

        return ee_poss, ee_oris, ee_vels, ee_omgs


    def get_jnt_states(self):

        jnt_poss = [ [] for _ in range (self._num_fingers)]
        jnt_vels = [ [] for _ in range (self._num_fingers)]
        jnt_reaction_forces =  [ [] for _ in range (self._num_fingers)]
        jnt_applied_torques  = [ [] for _ in range (self._num_fingers)]

        for finger_idx in range(self._num_fingers):

            for jnt_idx in self._finger_jnt_indices[finger_idx]:

                jnt_state = pb.getJointState(self._robot_id, jnt_idx)
                jnt_poss[finger_idx].append(jnt_state[0])
                jnt_vels[finger_idx].append(jnt_state[1])
                jnt_reaction_forces[finger_idx].append(jnt_state[2])
                jnt_applied_torques[finger_idx].append(jnt_state[3])


        return jnt_poss, jnt_vels, jnt_reaction_forces, jnt_applied_torques


    def get_mass_matrix(self, finger_idx, jnt_poss):

        jnt_pos = list(itertools.chain.from_iterable(jnt_poss))

        Mq = np.asarray(pb.calculateMassMatrix(bodyUniqueId=self._robot_id, objPositions=jnt_pos))

        #assuming equal finger joints
        active_joints = Mq.shape[1]/self._num_fingers
        
        return Mq[active_joints*finger_idx:active_joints*(finger_idx+1), 
                  active_joints*finger_idx:active_joints*(finger_idx+1)]


    def convert_fin_jnt_poss_to_list(self, fin_jnt_poss, only_mov_jnts=False):

        if only_mov_jnts:

            jnt_poss = np.zeros(len(self._jnt_indexs)).tolist()
        
        else:
            
            jnt_poss = np.zeros(self._tot_num_jnts).tolist()
        
        for k in range(self._num_fingers):

            for j, idx in enumerate(self._finger_jnt_indices[k]):

                jnt_poss[j] = fin_jnt_poss[k][j]

        return jnt_poss


    def get_jacobians(self, jnt_poss):

        jnt_pos = self.convert_fin_jnt_poss_to_list(jnt_poss)

        lin_jacs = []
        
        ang_jacs = []

        for k in range(self._num_fingers):

            linear_jacobian, angular_jacobian = pb.calculateJacobian(bodyUniqueId=self._robot_id,
                                                                linkIndex=self._ee_indexs[k],
                                                                localPosition=[0.,0.,0.],
                                                                objPositions=jnt_pos,
                                                                objVelocities=np.zeros(len(jnt_pos)).tolist(),
                                                                objAccelerations=np.zeros(len(jnt_pos)).tolist()
                                                                )
            lin_jacs.append(np.asarray(linear_jacobian))

        return lin_jacs


    def get_finger_jacobian(self, finger_idx, jnt_poss, local_point=None):

        if local_point is None:
            local_point = [0.,0.,0.]

        jnt_pos = self.convert_fin_jnt_poss_to_list(jnt_poss)

        linear_jacobian, angular_jacobian = pb.calculateJacobian(bodyUniqueId=self._robot_id,
                                                                linkIndex=self._ee_indexs[finger_idx],
                                                                localPosition=local_point,
                                                                objPositions=jnt_pos,
                                                                objVelocities=np.zeros(len(jnt_pos)).tolist(),
                                                                objAccelerations=np.zeros(len(jnt_pos)).tolist()
                                                                )

        Jee = np.asarray(linear_jacobian)
        
        cols = Jee.shape[1]/self._num_fingers

        return Jee[:, cols*finger_idx:cols*(finger_idx+1)]


    def get_hand_jacobian(self):

        jnt_poss = self.get_jnt_states()[0]

        jnt_pos = list(itertools.chain.from_iterable(jnt_poss))

        hand_jacobian = None

        for k in range(self._num_fingers):

            linear_jacobian, angular_jacobian = pb.calculateJacobian(bodyUniqueId=self._robot_id,
                                                                linkIndex=self._ee_indexs[k],
                                                                localPosition=[0.,0.,0.],
                                                                objPositions=jnt_pos,
                                                                objVelocities=np.zeros(self._num_joints).tolist(),
                                                                objAccelerations=np.zeros(self._num_joints).tolist()
                                                                )
            if hand_jacobian is None:
                
                hand_jacobian = np.asarray(linear_jacobian)
            
            else:
                
                hand_jacobian = np.vstack([hand_jacobian, np.asarray(linear_jacobian)])

        return hand_jacobian