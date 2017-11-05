#!/usr/bin/env python

import rospy
import intera_interface
import intera_external_devices
from intera_core_msgs.msg import SEAJointState

import time

def gravity_comp_callback(msg):
    print "name \n", msg.name
    print "commanded_effort \n", msg.commanded_effort
    print "commanded_velocity \n", msg.commanded_velocity
    print "commanded_position \n",msg.commanded_position
    print "actual_position \n", msg.actual_position
    print "actual_velocity \n", msg.actual_velocity
    print "actual_effort \n", msg.actual_effort
    print "gravity_model_effort \n", msg.gravity_model_effort

    torques = limb.joint_efforts()
    # print torques
    torques['right_j0'] = 0.01*msg.gravity_model_effort[0]
    torques['right_j1'] = 0.01*msg.gravity_model_effort[1]
    torques['right_j2'] = 0.01*msg.gravity_model_effort[2]
    torques['right_j3'] = 0.01*msg.gravity_model_effort[3]
    torques['right_j4'] = 0.01*msg.gravity_model_effort[4]
    torques['right_j5'] = 0.01*msg.gravity_model_effort[5]
    torques['right_j6'] = 0.01*msg.gravity_model_effort[6]
    limb.set_joint_torques(torques)

def listener():
    gravity_comp = rospy.Subscriber('robot/limb/right/gravity_compensation_torques',
                                    SEAJointState,  gravity_comp_callback)
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('gravity_compensation_test')
    limb = intera_interface.Limb('right')

    limb.exit_control_mode()
    limb.move_to_neutral()
    time.sleep(3)

    listener()

