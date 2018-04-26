# General ROS imports
import roslib

roslib.load_manifest('aml_robot')
import rospy

import numpy as np

# AML additional imports
from aml_robot.sawyer_robot import SawyerArm
from aml_io.log_utils import aml_logging


import threading

class SawyerPisaIIT(RobotInterface):
    def __init__(self, robot_name, on_state_callback=None):
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

