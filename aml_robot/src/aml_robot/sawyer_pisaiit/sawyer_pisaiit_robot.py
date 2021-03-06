# General ROS imports
import roslib

roslib.load_manifest('aml_robot')
import rospy

import numpy as np

import intera_interface

# AML additional imports
from aml_robot.robot_interface import RobotInterface
from aml_robot.sawyer_robot import SawyerArm
from aml_robot.pisaiit.pisaiit_robot import PisaIITHand
from aml_io.log_utils import aml_logging

import threading


class SawyerPisaIIT(RobotInterface):


    def __init__(self, robot_name='sawyer_pisa_iit', on_state_callback=None):
        """
        Class constructor
        Args: 
        robot_name: a string (ideally unique and human readable) representing this robot name
        on_state_callback: an optional callback
        Returns:
        none, store the trajectories
        """

        self._logger = aml_logging.get_logger(__name__)

        self._ready = False

        # Configuring hand (setting up publishers, variables, etc)
        self._configure(robot_name, on_state_callback)

        self._ready = True  # Hand is ready to be used

        self._lock = threading.Lock()
        self._state_update_thread = threading.Thread(target=self._update_state)
        self._state_update_thread.setDaemon(True)
        self._state_update_thread.start()

    def configure(self, robot_name, on_state_callback):

        self._arm = SawyerArm('right')
        self._hand = PisaIITHand('right')



        # def _on_current_status(self, msg):
        # self._logger.info(msg)
        #
        # self._sh_current_status = msg

    def _configure_cuff(self):

        self._has_cuff = True

        try:

            self._cuff_state = None

            self._cuff = intera_interface.Cuff(limb=self._arm._limb)
            # connect callback fns to signals
            self._lights = None
            self._lights = intera_interface.Lights()
            self._cuff.register_callback(self._light_action, '{0}_cuff'.format(self._arm._limb))




        except Exception as e:
            self._logger.warning(e)
            self._has_cuff = False

        if self._has_cuff:
            self._cuff.register_callback(self._close_action, '{0}_button_upper'.format(self._arm._limb))
            self._cuff.register_callback(self._open_action, '{0}_button_lower'.format(self._arm._limb))

    # this function returns self._cuff_state to be true
    # when arm is moved by a demonstrator, the moment arm stops
    # moving, the status returns to false
    # initial value of the cuff is None, it is made False by pressing the
    # cuff button once
    @property
    def get_lfd_status(self):
        if self._has_cuff:
            return self._cuff.cuff_button()
        else:
            self._logger.warning("CUFF NOT DETECTED")
            return None

    def _open_action(self, value):
        if value and self._gripper.is_ready():
            self._logger.debug("gripper open triggered")
            self._hand.exec_position_cmd([0.0])
            if self._lights:
                self._set_lights('red', False)
                self._set_lights('green', True)

    def _close_action(self, value):
        if value and self._gripper.is_ready():
            self._logger.debug("gripper close triggered")
            self._hand.exec_position_cmd([1.0])
            if self._lights:
                self._set_lights('green', False)
                self._set_lights('red', True)

    def _light_action(self, value):
        if value:
            self._logger.debug("cuff grasp triggered")
        else:
            self._logger.debug("cuff release triggered")
        if self._lights:
            self._set_lights('red', False)
            self._set_lights('green', False)
            self._set_lights('blue', value)

    def _set_lights(self, color, value):
        self._lights.set_light_state('head_{0}_light'.format(color), on=bool(value))
        self._lights.set_light_state('{0}_hand_{1}_light'.format(self._limb, color), on=bool(value))

    def cuff_cb(self, value):
        if self._has_cuff:
            self._cuff_state = value
        else:
            self._logger.warning("CUFF NOT DETECTED")

    def _update_state(self):
        self._logger.warning("Posi_update_state not implemented.")

    def _on_joint_states(self, msg):

        # Updates internal state

        if self._ready:
            self._state = self._update_state()
            self._on_state_callback(self._state)

    def exec_position_cmd(self, cmd):
        self._logger.warning("Position command delta not implemented.")

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

        self._logger.warning("forward_kinematics not implemented")

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
