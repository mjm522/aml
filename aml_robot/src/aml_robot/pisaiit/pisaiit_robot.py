import roslib
roslib.load_manifest('aml_robot')

import rospy

import numpy as np
import quaternion

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32

from aml_robot.robot_interface import RobotInterface


class PisaIITHand(RobotInterface):


    def __init__(self, robot_name, on_state_callback=None):
        """
        Class constructor
        Args: 
        robot_name: a string (ideally unique and human readable) representing this robot name
        on_state_callback: an optional callback
        Returns:
        none, store the trajectories
        """

        


        self._ready = False

        # Configuring hand (setting up publishers, variables, etc)
        self._configure(robot_name, on_state_callback)

        self._ready = True # Hand is ready to be used




    def _update_state(self):

        # now                 = rospy.Time.now()

        # joint_angles        = self.angles()
        # joint_velocities    = self.joint_velocities()
        # joint_efforts       = self.joint_efforts()

        # joint_names         = self.joint_names()

        # def to_list(ls):
        #     return [ls[n] for n in joint_names]

        state = {}
        # state['position']        = joint_angles
        # state['velocity']        = np.array(to_list(joint_velocities))
        # state['effort']          = np.array(to_list(joint_efforts))
        # state['jacobian']        = self.get_jacobian_from_joints(None)
        # state['inertia']         = self.get_inertia(None)
        # state['rgb_image']       = self._camera._curr_rgb_image
        # state['depth_image']     = self._camera._curr_depth_image
        # state['gravity_comp']    = np.array(self._h)


        # state['timestamp']       = { 'secs' : now.secs, 'nsecs': now.nsecs }

        # try:
        #     state['ee_point'], state['ee_ori']  = self.get_ee_pose()
        # except:
        #     pass

        # try:
        #     state['ee_vel'],   state['ee_omg']  = self.get_ee_velocity()
        # except:
        #     pass

        return state

    def _configure(self, limb, on_state_callback):
        self._state = None

        if on_state_callback:
            self._on_state_callback = on_state_callback
        else:
            self._on_state_callback = lambda m: None

        # Parent constructor
        # baxter_interface.limb.Limb.__init__(self, limb)

        #self._kinematics = baxter_kinematics(self)

        #self._ik_baxter = IKBaxter(limb=self)

        #self._ik_baxter.configure_ik_service()

        # self._pub_rate = rospy.Publisher('robot/joint_state_publish_rate',
        #                                  UInt16, queue_size=10)

        self._pos_cmd_pub = rospy.Publisher('soft_hand_pos_cmd', Float32, queue_size=10)
        self._sh_current_status = rospy.Publisher('soft_hand_read_current', Float32, queue_size=10)


        # self._camera = camera_sensor.CameraSensor()

        # self._ee_force = None
        # self._ee_torque = None


    def _on_joint_states(self, msg):
        
        # Updates internal state

        if self._ready:
            self._state = self._update_state()
            self._on_state_callback(self._state)


    def get_state(self):
        return self._state

    def get_end_effector_link_name(self):
        ''' todo '''
        pass

    
    def get_base_link_name(self):
        ''' todo '''
        pass

    
    def exec_position_cmd(self, cmd):
        ''' todo '''
        self._pos_cmd_pub.publish(float(cmd))

    def exec_position_cmd_delta(self,cmd):
        ''' todo '''
        pass

    def move_to_joint_pos_delta(self,cmd):
        ''' todo '''
        pass

    def move_to_joint_pos(self,cmd):
        ''' todo '''
        pass

    def exec_velocity_cmd(self,cmd):
        ''' todo '''
        pass

    def exec_torque_cmd(self,cmd):
        ''' todo '''
        pass

    def move_to_joint_position(self, joint_angles):
        ''' todo '''
        pass
    
    def get_ee_pose(self):
        ''' todo '''
        pass

    def get_time_in_seconds(self):
        time_now =  rospy.Time.now()
        return time_now.secs + time_now.nsecs*1e-9

    def get_ee_velocity(self, real_robot=True):
        ''' todo '''
        pass


    def get_cartesian_pos_from_joints(self, joint_angles=None):
        ''' todo '''

        pass
    def get_cartesian_vel_from_joints(self, joint_angles=None):
        ''' todo '''

        pass

    def get_jacobian_from_joints(self, joint_angles=None):
        ''' todo '''

        pass


    def get_inertia(self, joint_angles=None):
        ''' todo '''

        pass

    def set_speed(self,speed):
        ''' todo '''

        pass

    def ik(self, pos, ori=None):
        ''' todo '''

        pass