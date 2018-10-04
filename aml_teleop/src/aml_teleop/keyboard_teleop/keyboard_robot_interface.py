import rospy
import quaternion
import numpy as np
from aml_teleop.teleop import TeleOp
from aml_teleop.keyboard_teleop.config import KEYBOARD_ROBOT_CONFIG


class KeyboardRobotInterface(TeleOp):
    """
    base class of keyboard robot interface
    """
    def __init__(self, robot_interface, controller, config = KEYBOARD_ROBOT_CONFIG):

        TeleOp.__init__(self, "keyboard", robot_interface, config)

        self._ctrlr  = controller

        self._robot  = robot_interface

        self._config = config

        self._getch = Getch()


    def run(self):

        raise NotImplementedError("Must be implemented in the subclass")

    def get_robot_ee_pose(self):

        return self._robot._state['ee_point'], self._robot._state['ee_ori']


    def enable_ctrlr(self):
        """
        this function is for additional 
        safety of the device
        """

        if not self._ctrlr.is_active():

            self._ctrlr.set_active(True)

        else:

            self._ctrlr.set_active(False)


    def _teleoperate(self):

        c = self._getch()
        if c:
            #catch Esc or ctrl-c
            if c in ['\x1b', '\x03']:
                return True
                
            elif c in self._bindings:
                cmd = self._bindings[c]
                print("command: %s" % (cmd[2],))
                cmd[0](*cmd[1])

            elif c == '/' or '?':
                self._print_help()

            return False

    def _print_help(self):
        print("Key bindings: ")
        print("  Esc: Quit")
        print("  ?: Help")
        for key, val in sorted(self._bindings.items(),
                                           key=lambda x: x[1][2]):
                        print("  %s: %s" % (key, val[2]))

    def _run(self):
        """
        the main code that sends control commands to the
        robot
        """        
        rate = rospy.Rate(self._config['rate'])
        
        finished = False
        
        t = 0

        self.enable_ctrlr()
        while not rospy.is_shutdown() and not finished:

            finished = self._teleoperate()

            t = (t+1)

            rate.sleep()
        
        print("Exiting...")
        self._ctrlr.wait_until_goal_reached(timeout=5.0)
        
        self._ctrlr.set_active(False)


class Getch:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch