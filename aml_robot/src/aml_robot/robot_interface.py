import roslib
roslib.load_manifest('aml_robot')

import rospy
import numpy as np

class RobotInterface(object):

    def __init__(self, limb, on_state_callback=None):

        pass

    def set_sampling_rate(self, sampling_rate=100):
        """
        Not implemented, expected behaviour: sets update rate of state 
        """
        print "set_sampling_rate: NO EFFECT, NOT IMPLEMENTED"
        pass

    def _update_state(self):
    	pass

    def angles(self):
    	pass

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
    
    def get_ee_pose(self):
        """
        Not implemented
        """
        print "get_ee_pose: NO EFFECT, NOT IMPLEMENTED"
        pass

    def get_ee_velocity(self, real_robot=True):
        """
        Not implemented
        """
        print "get_ee_velocity: NO EFFECT, NOT IMPLEMENTED"
        pass

    def get_cartesian_pos_from_joints(self, joint_angles=None):
        """
        Not implemented
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
        Not implemented
        """
        print "get_jacobian_from_joints: NO EFFECT, NOT IMPLEMENTED"
        pass

    def get_inertia(self, joint_angles=None):
        """
        Not implemented
        """
        print "get_inertia: NO EFFECT, NOT IMPLEMENTED"
        pass

    def ik(self,position,orientation=None):
        """
        Not implemented
        """
        print "ik: NO EFFECT, NOT IMPLEMENTED"
        pass


    def joint_names():
        """
        Returns list of joint names for this robot
        """
        pass
        print "joint_names: NO EFFECT, NOT IMPLEMENTED"
        return []

    def joint_efforts():
        """
        Returns joint effort measurements values for this robot
        """
        pass
        print "joint_efforts: NO EFFECT, NOT IMPLEMENTED"
        return []


    def joint_velocities():
        """
        Returns joint velocities measurements values for this robot
        """
        pass
        print "joint_velocities: NO EFFECT, NOT IMPLEMENTED"
        return []