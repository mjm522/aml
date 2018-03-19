import roslib
roslib.load_manifest('aml_robot')

import rospy
import numpy as np

class RobotInterface(object):

    def __init__(self, limb, on_state_callback=None):

        pass

    """
    Robot setup methods
    """
    def set_sampling_rate(self, sampling_rate=100):
        """
        Not implemented, expected behaviour: sets update rate of state 
        """
        print "set_sampling_rate: NO EFFECT, NOT IMPLEMENTED"
        pass

    """
    Private methods
    """
    def _update_state(self):
    	pass


    """
    Command methods with side effects
    """

    def exec_position_cmd(self, cmd):
    	"""
        Not implemented
        """
        print "exec_position_cmd: NO EFFECT, NOT IMPLEMENTED"
        pass

    def exec_position_cmd_delta(self,cmd):
        """
        Not implemented
        """
        print "exec_position_cmd_delta: NO EFFECT, NOT IMPLEMENTED"
        pass

    def move_to_joint_pos_delta(self,cmd):
        """
        Not implemented
        """
        print "move_to_joint_pos_delta: NO EFFECT, NOT IMPLEMENTED"
        pass

    def move_to_joint_pos(self,cmd):
        """
        Not implemented
        """
        print "move_to_joint_pos: NO EFFECT, NOT IMPLEMENTED"
        pass 

    def exec_velocity_cmd(self,cmd):
        """
        Not implemented
        """
        print "exec_velocity_cmd: NO EFFECT, NOT IMPLEMENTED"
        pass 

    def exec_torque_cmd(self,cmd):
        """
        Not implemented
        """
        print "exec_torque_cmd: NO EFFECT, NOT IMPLEMENTED"
        pass 

    def move_to_joint_position(self, joint_angles):
        """
        Not implemented
        """
        print "move_to_joint_position: NO EFFECT, NOT IMPLEMENTED"
        pass


    def move_to_home(self):
        """
        Not implemented
        """
        print "move_to_home: NO EFFECT, NOT IMPLEMENTED"
        pass


    """
    Get methods
    """

    def q_mean(self):
        """
        Not implemented, should return the mean between joint limits
        """
        print "q_mean: NO EFFECT, NOT IMPLEMENTED"

    def n_cmd(self):
        """
        Not implemented, should return the number of control commands supported by this robot
        """
        print "n_cmd: NO EFFECT, NOT IMPLEMENTED"

    def n_joints(self):
        """
        Not implemented, should return the number of joints of this robot, may differ to n_cmd() 
        """
        print "n_joints: NO EFFECT, NOT IMPLEMENTED"

    def get_state(self):
        """
        Not implemented, should return the the state for this robot as a dictionary {'key': value}
        """
        print "get_state: NO EFFECT, NOT IMPLEMENTED"

    def angles(self):
        pass

    def joint_names(self):
        """
        Returns list of joint names for this robot
        """
        pass
        print "joint_names: NO EFFECT, NOT IMPLEMENTED"
        return []

    def joint_efforts(self):
        """
        Returns joint effort measurements values for this robot
        """
        pass
        print "joint_efforts: NO EFFECT, NOT IMPLEMENTED"
        return []


    def joint_velocities(self):
        """
        Returns joint velocities measurements values for this robot
        """
        pass
        print "joint_velocities: NO EFFECT, NOT IMPLEMENTED"
        return []
    
    def get_ee_pose(self):
        """
        Not implemented, should return end-effector pose
        """
        print "get_ee_pose: NO EFFECT, NOT IMPLEMENTED"
        pass

    def get_ee_velocity(self, real_robot=True):
        """
        Not implemented, should return end-effector velocity
        """
        print "get_ee_velocity: NO EFFECT, NOT IMPLEMENTED"
        pass

    def get_cartesian_pos_from_joints(self, joint_angles=None):
        """
        Not implemented, should return the forward kinematics over joint_angles
        """
        print "get_cartesian_pos_from_joints: NO EFFECT, NOT IMPLEMENTED"
        pass

    def get_cartesian_vel_from_joints(self, joint_angles=None):
        """
        Not implemented
        """
        print "get_cartesian_vel_from_joints: NO EFFECT, NOT IMPLEMENTED"
        pass

    def get_jacobian_from_joints(self, joint_angles=None):
        """
        Not implemented, should return the jacobian J(q) for the given joint angles
        """
        print "get_jacobian_from_joints: NO EFFECT, NOT IMPLEMENTED"
        pass

    def get_inertia(self, joint_angles=None):
        """
        Not implemented, should return the inertia tensor for the given joint angles M(q)
        """
        print "get_inertia: NO EFFECT, NOT IMPLEMENTED"
        pass

    def ik(self, position, orientation=None):
        """
        Not implemented, should attempt to solve inverse kinematics 
        for the given position and orientation
        """
        print "ik: NO EFFECT, NOT IMPLEMENTED"
        pass