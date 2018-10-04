#!/usr/bin/env python

import rospy
import argparse
import itertools
from aml_teleop.keyboard_teleop.config import OS_SAWYER_CONFIG
from aml_teleop.keyboard_teleop.config import OS_BAXTER_CONFIG
from aml_teleop.keyboard_teleop.os_teleop_ctrl import OSTeleopCtrl



def switch_baxter_arm():

    teleop.disable_ctrlr()
    print "Switching arm... Please wait..."
    teleop._robot = next(arm_switcher)
    teleop._ctrlr = teleop._ctrl_type(teleop._robot)
    teleop.enable_ctrlr()
    print "Arm switched!"


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Keyboard Teleop Task-Space Control Sawyer')

    parser.add_argument('-c', '--controller', type=str, default='pos', help='type of controller-(pos/torq/vel)')
    
    parser.add_argument('-r', '--robot', type=str, default='sawyer', help='arm interface, e.g. baxter/sawyer')

    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("test")

    if args.robot == 'sawyer':

        from aml_robot.sawyer_robot import SawyerArm

        r_limb = SawyerArm('right')
        r_limb.untuck()

        limbs = [r_limb]
        robot_config = OS_SAWYER_CONFIG

    elif args.robot == "baxter":

        from aml_robot.baxter_robot import BaxterArm

        r_limb = BaxterArm('right')
        l_limb = BaxterArm('left')
        r_limb.untuck()
        l_limb.untuck()

        limbs = [r_limb, l_limb]
        robot_config = OS_BAXTER_CONFIG

        robot_config['custom_controls'] = {'l': (switch_baxter_arm, [ ], "switch arm"),}

    else:
        raise Exception("Unknown Robot Interface.")

    robot_config['ctrlr_type'] = args.controller


    for limb in limbs:
        limb.set_arm_speed(max(robot_config['robot_max_speed'],robot_config['robot_min_speed'])) # WARNING: max 0.2 rad/s for safety reasons
        limb.set_sampling_rate(sampling_rate=200) # Arm should report its state as fast as possible.


    arm_switcher = itertools.cycle(limbs)

    print '\n',args,'\n'
    teleop = OSTeleopCtrl(next(arm_switcher),robot_config)
    teleop.run()


