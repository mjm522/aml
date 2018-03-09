import numpy as np
from aml_teleop.teleop import TeleOp
from aml_teleop.haptic_teleop.config import HAPTIC_ROBOT_CONFIG


class HapticRobotInterface(TeleOp):

    """
    base class of haptic robot interface
    """

    def __init__(self, haptic_interface, robot_interface, config = HAPTIC_ROBOT_CONFIG):

        TeleOp.__init__(self, haptic_interface, robot_interface, config)


    
