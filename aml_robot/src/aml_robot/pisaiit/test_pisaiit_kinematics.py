#!/usr/bin/env python

from aml_robot.pisaiit.pisaiit_kinematics import pisaiit_kinematics
from aml_robot.pisaiit.pisaiit_robot import PisaIITHand
from aml_io.io_tools import get_file_path, get_aml_package_path
import PyKDL
import numpy as np

import rospy



def test_njoints(hand_kinematics):

    print "Test n joints"
    for i in range(hand_kinematics.n_chains()):

        print hand_kinematics.n_joints(i)

def test_nlinks(hand_kinematics):

    print "Test n links"
    for i in range(hand_kinematics.n_chains()):

        print hand_kinematics.n_links(i)

def test_print_kdl_chain(hand_kinematics):

    for i in range(hand_kinematics.n_chains()):

        hand_kinematics.print_kdl_chain(i)

def test_jacobian(hand_kinematics):
    print "N chains: ", hand_kinematics.n_chains()
    for i in range(hand_kinematics.n_chains()):
        jacobian = hand_kinematics.jacobian(np.random.randn(hand_kinematics.n_joints(i)), finger_idx=i)

        print "Finger %d"%(i,), np.linalg.det(np.dot(jacobian.T,jacobian))


def test_forward_kinematics(hand_kinematics):


    for i in range(hand_kinematics.n_chains()):
        link_poses = hand_kinematics.forward_position_kinematics(np.random.randn(hand_kinematics.n_joints(i)), finger_idx=i)

        for j in range(len(link_poses)):
            print "Chain %d pose %d"%(i,j), link_poses[j]

        print "-------------------"


def main():
    rospy.init_node('pisa_iit_soft_hand_test', anonymous=True)

    pisaiit_hand = PisaIITHand()

    models_path = get_aml_package_path('aml_grasp/src/aml_grasp/models')
    hand_path = get_file_path('pisa_hand_right.urdf', models_path)
    hand_kinematics = pisaiit_kinematics(pisaiit_hand,hand_path)

    hand_kinematics.print_robot_description()

    test_forward_kinematics(hand_kinematics)
    # test_nlinks(hand_kinematics)
    # test_njoints(hand_kinematics)
    #
    test_print_kdl_chain(hand_kinematics)




if __name__ == '__main__':

    main()
