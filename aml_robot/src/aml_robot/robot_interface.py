import roslib
roslib.load_manifest('aml_robot')

import rospy

import baxter_interface
import baxter_external_devices

from std_msgs.msg import (
    UInt16,
)
from baxter_core_msgs.msg import SEAJointState
from baxter_interface import CHECK_VERSION
from baxter_kinematics import baxter_kinematics

import numpy as np
import quaternion

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


from aml_perception import camera_sensor 

#for computation of angular velocity
from aml_lfd.utilities.utilities import compute_omg

#from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES

class RobotInterface(object):

    def __init__(self, limb, on_state_callback=None):

        pass

    def set_sampling_rate(self, sampling_rate=100):
        pass

    def _update_state(self):
    	pass

    def angles(self):
    	pass

    def exec_position_cmd(self, cmd):
    	pass

    def exec_position_cmd2(self,cmd):
        pass

    def move_to_joint_pos_delta(self,cmd):
        pass

    def move_to_joint_pos(self,cmd):
        pass 

    def exec_velocity_cmd(self,cmd):
        
        pass

    def exec_torque_cmd(self,cmd):

        pass

    def move_to_joint_position(self, joint_angles):
        pass
    
    def get_ee_pose(self):
        
        pass

    def get_ee_velocity(self, real_robot=True):
        pass

    def get_cartesian_pos_from_joints(self, joint_angles=None):
        pass

    def get_cartesian_vel_from_joints(self, joint_angles=None):
        
        pass

    def get_jacobian_from_joints(self, joint_angles=None):
        
        pass

    def get_arm_inertia(self, joint_angles=None):
        pass

    def ik(self,position,orientation=None):
        pass