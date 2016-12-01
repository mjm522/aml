import roslib
roslib.load_manifest('aml_robot')

import rospy


import baxter_interface
import baxter_external_devices

from std_msgs.msg import (
    UInt16,
)

from baxter_interface import CHECK_VERSION
from baxter_kinematics import baxter_kinematics

import numpy as np
import quaternion

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


import camera_sensor   

#from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES

class BaxterArm(baxter_interface.limb.Limb):

    def __init__(self,limb,on_state_callback=None):

        self.ready = False

        self._configure(limb,on_state_callback)
        
        self.ready = True

    def _configure(self,limb,on_state_callback):
        self._state = None

        if on_state_callback:
            self._on_state_callback = on_state_callback
        else:
            self._on_state_callback = lambda m: None

        # Parent constructor
        baxter_interface.limb.Limb.__init__(self,limb)

        self._kinematics = baxter_kinematics(self)

        self._pub_rate = rospy.Publisher('robot/joint_state_publish_rate',
                                         UInt16, queue_size=10)

        self._pub_rate.publish(100)

        self.set_command_timeout(0.2)

        self._camera = camera_sensor.CameraSensor()

    def _update_state(self):

        joint_angles        = self.joint_angles()
        joint_velocities    = self.joint_velocities()
        joint_efforts       = self.joint_efforts()

        joint_names         = self.joint_names()

        def to_list(ls):
            return [ls[n] for n in joint_names]

        state = {}
        state['position']        = np.array(to_list(joint_angles))
        state['velocity']        = np.array(to_list(joint_velocities))
        state['effort']          = np.array(to_list(joint_efforts))
        state['jacobian']        = self.get_jacobian_from_joints(None)
        state['inertia']         = self.get_arm_inertia(None)
        state['rgb_image']       = self._camera.curr_rgb_image
        state['depth_image']     = self._camera.curr_depth_image

        try:
            state['ee_point'], state['ee_ori']  = self.get_ee_pose()
        except:
            pass

        # ee_velocity              = self.endpoint_velocity()['linear']
        # state['ee_velocity']     = np.array([ee_velocity.x, ee_velocity.y, ee_velocity.z])

        return state

    def angles(self):
        joint_angles        = self.joint_angles()

        joint_names         = self.joint_names()

        def to_list(ls):
            return [ls[n] for n in joint_names]

        return np.array(to_list(joint_angles))
        

    def _on_joint_states(self, msg):
        baxter_interface.limb.Limb._on_joint_states(self,msg)

        if self.ready:
            self._state = self._update_state()
            self._on_state_callback(self._state)


    def get_end_effector_link_name(self):
        return self._kinematics._tip_link

    
    def get_base_link_name(self):
        return self._kinematics._base_link

    
    def exec_position_cmd(self,cmd):

        joint_command = dict(zip(self.joint_names(), cmd))

        self.set_joint_positions(joint_command)

    def exec_position_cmd2(self,cmd):
        curr_q = self.joint_angles()
        joint_names = self.joint_names()
        joint_command = dict([(joint, curr_q[joint] + cmd[i]) for i, joint in enumerate(joint_names)])
        self.set_joint_positions(joint_command)

    def move_to_joint_pos_delta(self,cmd):
        curr_q = self.joint_angles()
        joint_names = self.joint_names()

        joint_command = dict([(joint, curr_q[joint] + cmd[i]) for i, joint in enumerate(joint_names)])

        self.move_to_joint_positions(joint_command)

    def move_to_joint_pos(self,cmd):
        curr_q = self.joint_angles()
        joint_names = self.joint_names()

        joint_command = dict([(joint, cmd[i]) for i, joint in enumerate(joint_names)])

        self.move_to_joint_positions(joint_command)

    def exec_velocity_cmd(self,cmd):
        
        joint_names = self.joint_names()

        #can't we use dict(zip(joint_names, cmd)) to combine two lists?
        velocity_command = dict([(joint, cmd[i]) for i, joint in enumerate(joint_names)])
        
        self.set_joint_velocities(velocity_command)

    def exec_torque_cmd(self,cmd):

        joint_names     = self.joint_names()

        torque_command  = dict([(joint, cmd[i]) for i, joint in enumerate(joint_names)])
        
        self.set_joint_torques(torque_command)

    def move_to_joint_position(self, joint_angles):
        self.move_to_joint_positions(dict(zip(self.joint_names(),joint_angles)))
    
    def get_ee_pose(self):
        
        ee_point        = self.endpoint_pose()['position']
        ee_point        = np.array([ee_point.x, ee_point.y, ee_point.z])
        
        ee_ori          = self.endpoint_pose()['orientation']
        ee_ori          = np.quaternion(ee_ori.w, ee_ori.x, ee_ori.y, ee_ori.z)
        
        return ee_point, ee_ori

    def get_ee_velocity(self):
        
        ee_velocity     = self.endpoint_velocity()['linear']
        ee_velocity     = np.array([ee_velocity.x, ee_velocity.y, ee_velocity.z])
        
        return ee_velocity


    def get_cartesian_pos_from_joints(self, joint_angles=None):
        
        if joint_angles is None:
            
            argument = None
        
        else:
            
            argument = dict(zip(self.joint_names(),joint_angles))
        
        #combine the names and joint angles to a dictionary, that only is accepted by kdl
        pose = np.array(self._kinematics.forward_position_kinematics(argument))
        position = pose[0:3][:,None] #senting as  column vector
        
        w = pose[6]; x = pose[3]; y = pose[4]; z = pose[5] #quarternions
        
        #formula for converting quarternion to rotation matrix

        rotation = np.array([[1.-2.*(y**2+z**2),    2.*(x*y-z*w),           2.*(x*z+y*w)],\
                             [2.*(x*y+z*w),         1.-2.*(x**2+z**2),      2.*(y*z-x*w)],\
                             [2.*(x*z-y*w),         2.*(y*z+x*w),           1.-2.*(x**2+y**2)]])
        
        return position, rotation

    def get_cartesian_vel_from_joints(self, joint_angles=None):
        
        if joint_angles is None:
            
            argument = None
        
        else:
            
            argument = dict(zip(self.joint_names(),joint_angles))
        
        #combine the names and joint angles to a dictionary, that only is accepted by kdl
        return np.array(self._kinematics.forward_velocity_kinematics(argument))[0:3] #only position

    def get_jacobian_from_joints(self, joint_angles=None):
        
        if joint_angles is None:
            
            argument = None
        
        else:
            
            argument = dict(zip(self.joint_names(),joint_angles))
        #combine the names and joint angles to a dictionary, that only is accepted by kdl
        return np.array(self._kinematics.jacobian(argument))


    def get_arm_inertia(self, joint_angles=None):
        if joint_angles is None:
            argument = None
        else:
            argument = dict(zip(self.joint_names(),joint_angles))

        return np.array(self._kinematics.inertia(argument))

