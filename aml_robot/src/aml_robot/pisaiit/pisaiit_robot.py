# General ROS imports
import roslib

roslib.load_manifest('aml_robot')
import rospy
from std_msgs.msg import Float32

import numpy as np
import quaternion

# AML additional imports
from aml_robot.robot_interface import RobotInterface
from aml_io.log_utils import aml_logging
from aml_robot.pisaiit.config import pisaiit_config


import threading


class PisaIITHand(RobotInterface):

    def __init__(self, robot_name = "right_hand", on_state_callback=None):
        """
        Class constructor
        Args: 
        robot_name: a string (ideally unique and human readable) representing this robot name
        on_state_callback: an optional callback
        Returns:
        none, store the trajectories
        """

        self._logger = aml_logging.get_logger(__name__,'critical')

        self._ready = False

        # Configuring hand (setting up publishers, variables, etc)
        self._configure(robot_name, on_state_callback)

        self._ready = True  # Hand is ready to be used

        self._lock = threading.Lock()
        self._state_update_thread = threading.Thread(target=self._update_state)
        self._state_update_thread.setDaemon(True)
        self._state_update_thread.start()




    def _update_state(self):

        while True:

            with self._lock:
                self._state = {}
                now                 = rospy.Time.now()
                self._state['timestamp'] = {'secs': now.secs, 'nsecs': now.nsecs}

                self._state['position']        = self.angles()
                self._state['velocity']        = self.joint_velocities()
                self._state['effort']          = self.joint_efforts()
                # state['jacobian']        = self.get_jacobian_from_joints(None)
                # state['inertia']         = self.get_inertia(None)
                # state['rgb_image']       = self._camera._curr_rgb_image
                # state['depth_image']     = self._camera._curr_depth_image
                # state['gravity_comp']    = np.array(self._h)




                # try:
                #     state['ee_point'], state['ee_ori']  = self.get_ee_pose()
                # except:
                #     pass

                # try:
                #     state['ee_vel'],   state['ee_omg']  = self.get_ee_velocity()
                # except:
                #     pass

                self._on_state_callback(self._state)

    def _configure(self, limb, on_state_callback):


        self.name = limb


        self._config = pisaiit_config

        self._links = self._config['links']

        self._joint_name_map = {"thumb": self._config['thumb_joints'],
                               "index": self._config['index_joints'],
                               "middle": self._config['middle_joints'],
                               "ring": self._config['ring_joints'],
                               "little": self._config['little_joints'],
                               "synergy": self._config['synergy_joints']}

        self._all_joint_names = []
        for finger_name in self._config["finger_order"]:
            self._all_joint_names += self._joint_name_map[finger_name]

        self._state = None

        if on_state_callback:
            self._on_state_callback = on_state_callback
        else:
            self._on_state_callback = lambda m: None

        self._pos_cmd_pub = rospy.Publisher('/soft_hand_pos_cmd', Float32, queue_size=10)
        self._sh_current_status = rospy.Subscriber('/soft_hand_read_current', Float32, callback=self._on_current_status, queue_size=10)
        self._nq = len(self._all_joint_names)
        self._nu = len(self._joint_name_map['synergy'])

        self._jnt_limits = [{'lower': 0.0, 'upper': 1.0}]
        self._q_mean = np.array([0.5 * (limit['lower'] + limit['upper']) for limit in self._jnt_limits])

    def _on_current_status(self, msg):
        self._logger.info(msg)

        self._sh_current_status = msg


    def _on_joint_states(self, msg):

        # Updates internal state

        if self._ready:
            self._state = self._update_state()
            self._on_state_callback(self._state)

    def exec_position_cmd(self, cmd):
        self._pos_cmd_pub.publish(cmd)

    def exec_position_cmd_delta(self, cmd):
        self._logger.warning("Position command delta not implemented.")

    def exec_velocity_cmd(self, cmd):
        self._logger.warning("Velocity commands not implemented.")

    def exec_torque_cmd(self, cmd):
        self._logger.warning("Torque commands not implemented.")

    def move_to_joint_position(self, cmd):
        self.exec_position_cmd(cmd)

    def move_to_joint_pos_delta(self, cmd):
        self._logger.warning("move_to_joint_pos_delta not implemented")

    def angles(self):
        self._logger.warning("angles not implemented")

        return np.zeros(self.n_joints())

    def joint_velocities(self):
        self._logger.warning("joint_velocities not implemented")

        return np.zeros(self.n_joints())

    def joint_efforts(self):
        self._logger.warning("joint_efforts not implemented")

        return np.zeros(self.n_joints())

    def ee_velocity(self, numerical=False):
        self._logger.warning("ee_velocity not implemented")

    def q_mean(self):
        return self._q_mean

    def inertia(self, joint_angles=None):
        self._logger.warning("inertia not implemented")

    def cartesian_velocity(self, joint_velocities=None):
        self._logger.warning("cartesian_velocity not implemented")

    def forward_kinematics(self, joint_angles=None, ori_type='quat'):

        if joint_angles is None:

            argument = None

        else:

            argument = dict(zip(self.joint_names(), joint_angles))

        # combine the names and joint angles to a dictionary, that only is accepted by kdl
        pose = np.array(self._kinematics.forward_position_kinematics(argument))
        position = pose[0:3][:, None]  # senting as  column vector

        w = pose[6]
        x = pose[3]
        y = pose[4]
        z = pose[5]  # quarternions

        rotation = quaternion.quaternion(w, x, y, z)

        # formula for converting quarternion to rotation matrix

        if ori_type == 'mat':

            # rotation = np.array([[1.-2.*(y**2+z**2),    2.*(x*y-z*w),           2.*(x*z+y*w)],\
            #                      [2.*(x*y+z*w),         1.-2.*(x**2+z**2),      2.*(y*z-x*w)],\
            #                      [2.*(x*z-y*w),         2.*(y*z+x*w),           1.-2.*(x**2+y**2)]])

            rotation = quaternion.as_rotation_matrix(rotation)

        elif ori_type == 'eul':

            rotation = quaternion.as_euler_angles(rotation)
        elif ori_type == 'quat':
            pass

        return position, rotation

    def inverse_kinematics(self, position, orientation=None):
        self._logger.warning("inverse_kinematics not implemented")

    def n_cmd(self):
        return self._nu

    def n_joints(self):
        return self._nq


    def tuck(self):
        self._logger.warning("tuck not implemented")

    def untuck(self):
        self._logger.warning("untuck not implemented")

    def joint_names(self):
        return self._all_joint_names

    def links(self):
        return self._links

    def joint_limits(self):
        return self._jnt_limits

    def jacobian(self, joint_angles=None):
        self._logger.warning("jacobian not implemented")
        return 0.0

    def state(self):
        with self._lock:
            return self._state

    def ee_pose(self):
        self._logger.warning("ee_pose commands not implemented.")

    def set_sampling_rate(self, sampling_rate=100):
        self._logger.warning("set_sampling_rate commands not implemented.")
