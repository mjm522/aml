#!/usr/bin/env python

import rospy
import intera_interface
from intera_interface import CHECK_VERSION
from intera_interface import Gripper
import rospkg
import random
import numpy as np

class InteraSDKTest(object):
    def __init__(self, limb="right", display=True):
        self._limb = intera_interface.Limb(limb)
        self._head = intera_interface.Head()
        try:
            self._gripper = intera_interface.Gripper(limb)
        except ValueError:
            self._has_gripper = False
            rospy.logerr("Could not detect a gripper attached to the robot.")
        except ValueError:
            self._has_gripper = True

        if display == True:
            head_display = intera_interface.HeadDisplay()
            rospack = rospkg.RosPack()
            images_dir = rospack.get_path('intera_examples') + '/share/images/'
            head_display.display_image(images_dir + "sawyer_sdk_research.png", False, 1.0)

        self._joint_names = self._limb.joint_names()
        # print (self._joint_names)

        print("Getting robot state... ")
        self._rs = intera_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        print("Running. Ctrl-c to quit")

    def move_to_neutral(self):
        self._limb.move_to_neutral()
        self._head.set_pan(0.0)
        self._gripper.open()

    def wobble(self):
        self.move_to_neutral()
        command_rate = rospy.Rate(1)
        control_rate = rospy.Rate(100)
        start = rospy.get_time()

        current_joints = self._limb.joint_angles()
        desired_joints = self._limb.joint_angles()

        while not rospy.is_shutdown() and (rospy.get_time() - start < 10.0):
            angle = random.uniform(-2.0, 0.95)
            while (not rospy.is_shutdown() and
                   not (abs(self._head.pan() - angle) <=
                       intera_interface.HEAD_PAN_ANGLE_TOLERANCE)):
                self._head.set_pan(angle, speed=0.3, timeout=0)

                for name in self._joint_names:
                    desired_joints[name] = current_joints[name] + random.uniform(-0.1, 0.1)
                self._limb.set_joint_positions(desired_joints)

                desired_grip = random.uniform(Gripper.MIN_POSITION, Gripper.MAX_POSITION)
                self._gripper.set_position(desired_grip)

                control_rate.sleep()
            command_rate.sleep()

    def clean_shutdown(self):
        print("\nExiting example...")
        self.move_to_neutral()
        self._limb.exit_control_mode()

def main():
    print("Initializing node... ")
    rospy.init_node("intera_sdk_test")

    ist = InteraSDKTest()
    rospy.on_shutdown(ist.clean_shutdown)
    ist.wobble()

if __name__ == "__main__":
    main()
