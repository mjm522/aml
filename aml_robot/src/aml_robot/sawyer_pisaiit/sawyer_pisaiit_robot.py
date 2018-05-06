# General ROS imports
import roslib

roslib.load_manifest('aml_robot')
import rospy

import numpy as np

# AML additional imports
from aml_robot.sawyer_robot import SawyerArm
from aml_robot.pisaiit.pisaiit_robot import PisaIITHand
from aml_io.log_utils import aml_logging



import threading

class SawyerPisaIIT(RobotInterface):
    def __init__(self, robot_name = 'sawyer_pisa_iit', on_state_callback=None):
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

    def configure(self,robot_name, on_state_callback):

        self._arm = SawyerArm('right')
        self._hand = PisaIITHand('right')



        def _on_current_status(self, msg):
        self._logger.info(msg)

        self._sh_current_status = msg


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




