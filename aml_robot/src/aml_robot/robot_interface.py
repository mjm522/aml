import roslib
roslib.load_manifest('aml_robot')

import abc

class RobotInterface(object):
    __metaclass__ = abc.ABCMeta

    """
    Robot setup methods
    """

    @abc.abstractmethod
    def set_sampling_rate(self, sampling_rate=100):
        """
        Not implemented, expected behaviour: sets update rate of state 
        """
        raise NotImplementedError("set_sampling_rate: NO EFFECT, NOT IMPLEMENTED")

    """
    Private methods
    """

    @abc.abstractmethod
    def _update_state(self):
        """
        Not implemented
        """
        raise NotImplementedError("_update_state: NO EFFECT, NOT IMPLEMENTED")


    """
    Command methods with side effects
    """

    @abc.abstractmethod
    def exec_position_cmd(self, cmd):
    	"""
        Not implemented
        """
        raise NotImplementedError("exec_position_cmd: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def exec_position_cmd_delta(self,cmd):
        """
        Not implemented
        """
        raise NotImplementedError("exec_position_cmd_delta: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def move_to_joint_pos_delta(self,cmd):
        """
        Not implemented
        """
        raise NotImplementedError("move_to_joint_pos_delta: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def move_to_joint_position(self,cmd):
        """
        Not implemented
        """
        raise NotImplementedError("move_to_joint_pos: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def exec_velocity_cmd(self,cmd):
        """
        Not implemented
        """
        raise NotImplementedError("exec_velocity_cmd: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def exec_torque_cmd(self,cmd):
        """
        Not implemented
        """
        raise NotImplementedError("exec_torque_cmd: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def untuck(self):
        """
        Not implemented
        """
        raise NotImplementedError("move_to_home: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def tuck(self):
        """
        Not implemented
        """
        raise NotImplementedError("move_to_home: NO EFFECT, NOT IMPLEMENTED")


    """
    Get methods
    """

    @abc.abstractproperty
    def q_mean(self):
        """
        Not implemented, should return the mean between joint limits
        """
        raise NotImplementedError("q_mean: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractproperty
    def n_cmd(self):
        """
        Not implemented, should return the number of control commands supported by this robot
        """
        raise NotImplementedError("n_cmd: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractproperty
    def n_joints(self):
        """
        Not implemented, should return the number of joints of this robot, may differ to n_cmd() 
        """
        raise NotImplementedError("n_joints: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractproperty
    def state(self):
        """
        Not implemented, should return the the state for this robot as a dictionary {'key': value}
        """
        raise NotImplementedError("get_state: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractproperty
    def angles(self):
        """
        Returns current joint angles measured by encoders
        """
        raise NotImplementedError("angles: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractproperty
    def joint_names(self):
        """
        Returns list of joint names for this robot
        """
        raise NotImplementedError("joint_names: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractproperty
    def joint_efforts(self):
        """
        Returns joint effort measurements values for this robot
        """
        raise NotImplementedError("joint_efforts: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractproperty
    def joint_velocities(self):
        """
        Returns joint velocities measurements values for this robot
        """
        raise NotImplementedError("joint_velocities: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractproperty
    def ee_pose(self):
        """
        Not implemented, should return end-effector pose
        """
        raise NotImplementedError("get_ee_pose: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def ee_velocity(self, numerical=False):
        """
        Not implemented, should return end-effector velocity
        """
        raise NotImplementedError("get_ee_velocity: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def forward_kinematics(self, joint_angles=None):
        """
        Not implemented, should return the forward kinematics over joint_angles
        """
        raise NotImplementedError("get_cartesian_pos_from_joints: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def inverse_kinematics(self, position, orientation=None):
        """
        Not implemented, should attempt to solve inverse kinematics 
        for the given position and orientation
        """
        raise NotImplementedError("inverse_kinematics: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def cartesian_velocity(self, joint_velocities=None):
        """
        Not implemented
        """
        # xdot = J*qdot
        raise NotImplementedError("cartesian_velocity: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def jacobian(self, joint_angles=None):
        """
        Not implemented, should return the jacobian J(q) for the given joint angles
        """
        raise NotImplementedError("jacobian: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def inertia(self, joint_angles=None):
        """
        Not implemented, should return the inertia tensor for the given joint angles M(q)
        """
        raise NotImplementedError("inertia: NO EFFECT, NOT IMPLEMENTED")
