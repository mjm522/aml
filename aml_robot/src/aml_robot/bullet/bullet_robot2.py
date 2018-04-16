import numpy as np
import quaternion

from aml_math.quaternion_utils import compute_omg

import pybullet as pb


class BulletRobot2(object):
    def __init__(self, robot_id, config):

        """
        :param config: configuration for this robot
        """

        self._id = robot_id

        # self._config = config
        # description_path = config['description_path']
        # extension = description_path.split('.')[-1]
        # if extension == "urdf":
        #     self._id = pb.loadURDF(config['description_path'])
        # elif extension == "xml":
        #     self._id = pb.loadMJCF(config['description_path'])[0]

        self._all_joints = range(pb.getNumJoints(self._id))

        self._movable_joints = self.get_movable_joints()

        self._nq = len(self._all_joints)

        self._nu = len(self._movable_joints)

        joint_information = self.get_joint_info()

        self._all_joint_names = [info['jointName'] for info in joint_information]

        self._all_joint_dict = dict(zip(self._all_joint_names, self._all_joints))

        self._ee_link_name = config['ee_link_name']#ee_link_name  # config['ee_link_name']

        self._ee_link_idx = config['ee_link_idx']#ee_link_idx  # config['ee_link_idx'] # saywer ee idx: 16

        self._joint_limits = self.get_joint_limits()

    def state(self):
        """
        
        :return: returns dictionary containing current state of the robot
        """

        joint_angles = self.angles()
        joint_velocities = self.joint_velocities()
        joint_efforts = self.joint_efforts()

        state = {}
        state['position'] = joint_angles
        state['velocity'] = joint_velocities
        state['effort'] = joint_efforts
        state['jacobian'] = self.jacobian(None)
        state['inertia'] = self.inertia(None)

        state['ee_point'], state['ee_ori'] = self.ee_pose()

        state['ee_vel'], state['ee_omg'] = self.ee_velocity()

        # state['gripper_state'] = self.gripper_state()

    def jacobian(self, joint_angles=None):
        """
        
        :param joint_angles: 
        Optional parameter. If different than None, then returned jacobian will be evaluated at given joint_angles.
        Otherwise the returned jacobian will be evaluated at current robot joint angles
        
        :return: 
        Jacobian evaluated at current joint angles or optionally at joint_angles
        
        """

        if joint_angles is None:
            joint_angles = self.angles()

        linear_jacobian, angular_jacobian = pb.calculateJacobian(bodyUniqueId=self._id,
                                                                 linkIndex=self._ee_link_idx,
                                                                 localPosition=[0.0, 0.0, 0.0],
                                                                 objPositions=joint_angles.tolist(),
                                                                 objVelocities=np.zeros(self.n_joints()).tolist(),
                                                                 objAccelerations=np.zeros(self.n_joints()).tolist()
                                                                 )

        jacobian = np.vstack([np.array(linear_jacobian), np.array(angular_jacobian)])

        return jacobian

    def ee_pose(self):
        """
        
        :return: end-effector pose of this robot in the format (position,orientation)
        Note: orientation is a quaternion following Hamilton convention, i.e. (w, x, y, z)
        """
        return self.get_link_pose(link_id=self._ee_link_idx)

    def ee_velocity(self, numerical=False):
        """
        
        :param numerical: flag indicating if end-effector velocity should be computed numerically or not.
        :return: end-effector velocity, which includes linear and angular velocities, i.e. (v,omega)
        """

        return self.get_link_velocity(link_id=self._ee_link_idx)

    def inertia(self, joint_angles=None):
        """
        
        :param joint_angles: optional parameter, if not None, then returned inertia is evaluated at given joint_angles.
        Otherwise, returned inertia tensor is evaluated at current joint angles.
        :return: Joint space inertia tensor
        """

        if joint_angles is None:
            joint_angles = self.angles()

        inertia_tensor = np.array(pb.calculateMassMatrix(self._id, joint_angles.tolist()))

        return inertia_tensor

    def inverse_kinematics(self, position, orientation=None):
        """
        
        :param position: target position
        :param orientation: target orientation in quaternion format (w, x, y , z)
        :return: joint positions that take the end effector to the desired target position and/or orientation
        """

        solution = None
        if orientation is None:

            solution = pb.calculateInverseKinematics(self._id,
                                                     self._ee_link_idx,
                                                     targetPosition=position)
        else:

            orientation = [orientation[1], orientation[2], orientation[3], orientation[0]]
            solution = pb.calculateInverseKinematics(self._id,
                                                     self._ee_link_idx,
                                                     targetPosition=position,
                                                     targetOrientation=orientation)

        return np.array(solution), solution is None

    def q_mean(self):
        return self._joint_limits['mean']

    def joint_names(self):
        """
        
        :return: joint names for this robot
        """
        return self._all_joint_names

    def joint_ids(self):

        return self._all_joints

    def angles(self):

        """
        
        :return: current joint angles
        """
        return self.get_joint_state()[0]

    def joint_velocities(self):

        """
        
        
        :return: current joint velocities
        """
        return self.get_joint_state()[1]

    def joint_efforts(self):
        """
        
        
        :return: current joint efforts
        """
        return self.get_joint_state()[3]

    def n_joints(self):
        """
        
        :return: number of joint this robot has
        """
        return self._nq

    def n_cmd(self):
        """
        
        
        :return: number of controllable degrees of freedom that this robot has
        """
        return self._nu

    def set_joint_velocities(self, cmd, joints=None):

        if joints is None:
            joints = self._movable_joints

        pb.setJointMotorControlArray(self._id, joints, controlMode=pb.VELOCITY_CONTROL, targetVelocities=cmd)

    def set_joint_torques(self, cmd, joints=None):

        if joints is None:
            joints = self._movable_joints

        pb.setJointMotorControlArray(self._id, joints, controlMode=pb.TORQUE_CONTROL, forces=cmd)

    def set_joint_positions_delta(self, cmd, joints=None, forces=None):

        if joints is None:
            joints = self._movable_joints

        if forces is None:
            forces = np.ones(len(joints)) * 1.5

        pb.setJointMotorControlArray(self._id, joints,
                                     controlMode=pb.POSITION_CONTROL,
                                     targetVelocities=cmd,
                                     forces=forces)

    def set_joint_positions(self, cmd, joints=None, forces=None):

        vels = [0.0005 for n in range(len(cmd))]
        pb.setJointMotorControlArray(self._id, joints, controlMode=pb.POSITION_CONTROL,
                                     targetPositions=cmd, targetVelocities=vels)

    def get_joint_info(self):
        attribute_list = ['jointIndex', 'jointName', 'jointType',
                          'qIndex', 'uIndex', 'flags',
                          'jointDamping', 'jointFriction', 'jointLowerLimit',
                          'jointUpperLimit', 'jointMaxForce', 'jointMaxVelocity', 'linkName']

        joint_information = []
        for idx in self._all_joints:
            info = pb.getJointInfo(self._id, idx)
            joint_information.append(dict(zip(attribute_list, info)))

        return joint_information

    def get_link_pose(self, link_id=-3):

        if link_id == -3:
            self._ee_link_idx

        # if link_id not in self._joint_idx:
        #     raise Exception("Invalid Link ID")

        link_state = pb.getLinkState(self._id, link_id)
        pos = np.asarray(link_state[0])
        ori = np.quaternion(link_state[1][3], link_state[1][0], link_state[1][1],
                            link_state[1][2])  # hamilton convention

        return pos, ori

    def get_link_velocity(self, link_id=-3):

        if link_id == -3:
            self._ee_link_idx

        # if link_id not in self._joint_idx:
        #     raise Exception("Invalid Link ID")

        link_state = pb.getLinkState(self._id, link_id, computeLinkVelocity=1)

        lin_vel = np.asarray(link_state[6])
        ang_vel = np.asarray(link_state[7])

        return lin_vel, ang_vel

    def get_joint_state(self):

        joint_angles = []
        joint_velocities = []
        joint_reaction_forces = []
        joint_efforts = []

        for idx in self._all_joints:
            joint_state = pb.getJointState(self._id, idx)

            joint_angles.append(joint_state[0])

            joint_velocities.append(joint_state[1])

            joint_reaction_forces.append(joint_state[2])

            joint_efforts.append(joint_state[3])

        return np.array(joint_angles), np.array(joint_velocities), np.array(joint_reaction_forces), np.array(
            joint_efforts)

    def set_jnt_state(self, jnt_state):

        num_jnts = pb.getNumJoints(self._id)

        if len(jnt_state) < num_jnts:
            raise Exception ("Incorrect number of joint state values given")

        else:
            for jnt_idx in range(num_jnts):

                pb.resetJointState(self._id, jnt_idx, jnt_state[jnt_idx])

    def get_movable_joints(self):

        movable_joints = []
        for i in self._all_joints:
            joint_info = pb.getJointInfo(self._id, i)
            q_index = joint_info[3]
            joint_name = joint_info[1]
            if q_index > -1:
                movable_joints.append(i)

        return np.array(movable_joints)

    def get_all_joints(self):

        return np.array(self._all_joints)

    def get_joint_dict(self):

        return self._all_joint_dict

    def get_joint_limits(self):

        lower_lim = np.zeros(self.n_joints())

        upper_lim = np.zeros(self.n_joints())

        mean_ = np.zeros(self.n_joints())

        range_ = np.zeros(self.n_joints())

        for k, idx in enumerate(self._all_joints):
            lower_lim[k] = pb.getJointInfo(self._id, idx)[8]

            upper_lim[k] = pb.getJointInfo(self._id, idx)[9]

            mean_[k] = 0.5 * (lower_lim[k] + upper_lim[k])

            range_[k] = (upper_lim[k] - lower_lim[k])

        # [dict([x]) for x in zip(['upper'] * a.n_cmd(), a._bullet_robot.get_joint_limits()['upper'])]
        #
        return {'lower': lower_lim, 'upper': upper_lim, 'mean': mean_, 'range': range_}

    def get_joint_by_name(self, joint_name):
        if joint_name in self._all_joint_dict:
            return self._all_joint_dict[joint_name]
        else:
            raise Exception("Joint name does not exist!")


    def configure_default_pos(self, pos, ori):

        self._default_pos = pos
        self._default_ori = ori
        self.set_default_pos_ori()

    def set_default_pos_ori(self):

        self.set_pos_ori(self._default_pos, self._default_ori)

    def set_pos_ori(self, pos, ori):

        pb.resetBasePositionAndOrientation(self._id, pos, ori)

    def set_ctrl_mode(self, ctrl_type='pos'):

        angles = self.angles()

        # disable the default position_control mode.
        for k, jnt_index in enumerate(self._movable_joints):

            pb.resetJointState(self._id, jnt_index, angles[k])

            if ctrl_type == 'pos':

                pb.setJointMotorControl2(self._id, jnt_index, pb.POSITION_CONTROL,
                                         targetPosition=angles[k], force=10)

            else:

                pb.setJointMotorControl2(self._robot_id, jnt_index, pb.VELOCITY_CONTROL,
                                         targetPosition=angles[k], force=0.5)

    # def get_image(self):
    #
    #     _, _, self._view_matrix, _, _, _, _, _, _, _, _, _ = pb.getDebugVisualizerCamera()
    #     (w, h, rgb_pixels, depth_pixels, _) = pb.getCameraImage(self._config['cam']['image_width'],
    #                                                             self._config['cam']['image_height'],
    #                                                             self._view_matrix, self._projection_matrix, [0, 1, 0],
    #                                                             renderer=pb.ER_BULLET_HARDWARE_OPENG)
    #
    #     # rgba to rgb
    #     rgb_image = rgb_pixels[:, :, 0:3]
    #     depth_image = depth_pixels
    #
    #     return rgb_image, depth_image


