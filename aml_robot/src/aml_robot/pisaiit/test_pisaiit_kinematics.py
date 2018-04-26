#!/usr/bin/env python

from aml_robot.pisaiit.pisaiit_kinematics import pisaiit_kinematics
from aml_robot.pisaiit.pisaiit_robot import PisaIITHand
from aml_io.io_tools import get_file_path, get_aml_package_path
import PyKDL
import numpy as np

import rospy


def main():
    rospy.init_node('pisa_iit_soft_hand_test', anonymous=True)

    pisaiit_hand = PisaIITHand()

    models_path = get_aml_package_path('aml_grasp/src/aml_grasp/models')
    hand_path = get_file_path('pisa_hand_right.urdf', models_path)
    hand_kinematics = pisaiit_kinematics(pisaiit_hand,hand_path)

    hand_kinematics.print_robot_description()

    for i in range(hand_kinematics._num_chains):

        hand_kinematics.print_kdl_chain(i)

    finger_idx = 1
    num_joints = 7#hand_kinematics._chains[finger_idx].getNrOfJoints()
    jacobian = PyKDL.Jacobian(num_joints)

    for i in range(1):
        kdl_array = PyKDL.JntArray(num_joints)
        angles = np.random.randn(num_joints)
        for idx in range(angles.shape[0]):
            kdl_array[idx] = angles[idx]
        hand_kinematics._jac_kdl[finger_idx].JntToJac(kdl_array, jacobian)

        print hand_kinematics.kdl_to_mat(jacobian)



if __name__ == '__main__':

    main()
