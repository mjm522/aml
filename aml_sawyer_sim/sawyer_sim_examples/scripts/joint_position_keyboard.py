#!/usr/bin/env python

import argparse
import rospy
import intera_interface
import intera_external_devices
from intera_interface import CHECK_VERSION


def map_keyboard(side='right'):
    limb = intera_interface.Limb(side)
    limb.move_to_neutral()
    head = intera_interface.Head()
    head.set_pan(0.0)

    try:
        gripper = intera_interface.Gripper(side)
    except:
        has_gripper = False
        rospy.logerr("Could not initalize the gripper.")
    else:
        has_gripper = True

    joints = limb.joint_names()

    def set_j(limb, joint_name, delta):
        current_position = limb.joint_angle(joint_name)
        joint_command = {joint_name: current_position + delta}
        limb.set_joint_positions(joint_command)

    def offset_position(offset):
        current = gripper.get_position()
        gripper.set_position(current + offset)

    num_steps = 10.0

    bindings = {
        '1': (set_j, [limb, joints[0], 0.1], joints[0]+" increase"),
        'q': (set_j, [limb, joints[0], -0.1], joints[0]+" decrease"),
        '2': (set_j, [limb, joints[1], 0.1], joints[1]+" increase"),
        'w': (set_j, [limb, joints[1], -0.1], joints[1]+" decrease"),
        '3': (set_j, [limb, joints[2], 0.1], joints[2]+" increase"),
        'e': (set_j, [limb, joints[2], -0.1], joints[2]+" decrease"),
        '4': (set_j, [limb, joints[3], 0.1], joints[3]+" increase"),
        'r': (set_j, [limb, joints[3], -0.1], joints[3]+" decrease"),
        '5': (set_j, [limb, joints[4], 0.1], joints[4]+" increase"),
        't': (set_j, [limb, joints[4], -0.1], joints[4]+" decrease"),
        '6': (set_j, [limb, joints[5], 0.1], joints[5]+" increase"),
        'y': (set_j, [limb, joints[5], -0.1], joints[5]+" decrease"),
        '7': (set_j, [limb, joints[6], 0.1], joints[6]+" increase"),
        'u': (set_j, [limb, joints[6], -0.1], joints[6]+" decrease"),
        '8': (gripper.close, [], "gripper close"),
        'i': (gripper.open, [], "gripper open"),
        '9': (offset_position, [-(gripper.MAX_POSITION / num_steps)], "gripper decrease"),
        'o': (offset_position, [gripper.MAX_POSITION / num_steps], "gripper increase"),
        '0': (head.set_pan, [head.pan() - 0.1], "head pan decrease"),
        'p': (head.set_pan, [head.pan() + 0.1], "head pan increase"),
     }
    done = False
    print("Controlling joints. Press ? for help, Esc to quit.")
    while not done and not rospy.is_shutdown():
        c = intera_external_devices.getch()
        if c:
            #catch Esc or ctrl-c
            if c in ['\x1b', '\x03']:
                done = True
                rospy.signal_shutdown("Example finished.")
            elif c in bindings:
                cmd = bindings[c]
                cmd[0](*cmd[1])
                print("command: %s" % (cmd[2],))
            else:
                print("key bindings: ")
                print("  Esc: Quit")
                print("  ?: Help")
                for key, val in sorted(bindings.items(),
                                       key=lambda x: x[1][2]):
                    print("  %s: %s" % (key, val[2]))

def main():
    print("Initializing node... ")
    rospy.init_node("joint_position_keyboard")

    print("Getting robot state... ")
    rs = intera_interface.RobotEnable(CHECK_VERSION)
    init_state = rs.state().enabled

    def clean_shutdown():
        print("\nExiting example.")

    rospy.on_shutdown(clean_shutdown)

    rospy.loginfo("Enabling robot...")
    rs.enable()

    map_keyboard()

    print("Done.")

if __name__ == '__main__':
    main()
