import numpy as np
from aml_teleop.teleop import TeleOp
from aml_teleop.haptic_teleop.config import HAPTIC_ROBOT_CONFIG


class HapticRobotInterface(TeleOp):
    """
    base class of haptic robot interface
    """
    def __init__(self, haptic_interface, robot_interface, controller, config = HAPTIC_ROBOT_CONFIG):

        TeleOp.__init__(self, haptic_interface, robot_interface, config)

        self._ctrlr  = controller

        self._haptic = haptic_interface

        self._robot  = robot_interface

    def run(self):

        raise NotImplementedError("Must be implemented in the subclass")

    def enable_ctrlr(self):
        """
        this function is for additional 
        safety of the device
        """

        if self._haptic._device_enabled:

            self._ctrlr.set_active(True)

        else:

            self._ctrlr.set_active(False)