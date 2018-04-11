#!/usr/bin/env python

import numpy as np
import pybullet as pb
import cv2
import rospy
from aml_robot.bullet.bullet_sawyer import BulletSawyerArm
from aml_io.io_tools import get_aml_package_path, get_abs_path
# from matplotlib import pyplot as plt


def main():

    rospy.init_node('poke_box', anonymous=True)

    timeStep = 0.01
    
    phys_id = pb.connect(pb.SHARED_MEMORY)

    
    if (phys_id<0):
        phys_id = pb.connect(pb.GUI)
    
    pb.resetSimulation()
    
    pb.setTimeStep(timeStep)

    pb.setGravity(0., 0.,-10.0)


    catkin_ws_src_path = get_abs_path(get_aml_package_path()+'/../..')
    print "catkin_ws_src_path:", catkin_ws_src_path
    # "/sawyer_robot/sawyer_description/urdf/sawyer.urdf"
    manipulator = pb.loadURDF(catkin_ws_src_path + '/src/aml/aml_rl/aml_rl_envs/src/aml_rl_envs/models/sawyer/sawyer2_with_pisa_hand.urdf', useFixedBase=True)
    # pb.resetBasePositionAndOrientation(manipulator,[0,0,0],[0,0,0,1])
    # motors = [n for n in range(pb.getNumJoints(manipulator))]

    arm = BulletSawyerArm(manipulator)


    rate = rospy.Rate(500)

    pb.setRealTimeSimulation(1)


    # import time

    # time.sleep(1)
    # sawyerArm.exec_velocity_cmd([0.5,0,0,0,0,0,0])

    # rospy.on_shutdown(self.on_shutdown)

    # self._record_sample.start_record(task_action=pushes[idx])
    # plt.ion()

    arm._bullet_robot.set_ctrl_mode('pos')
    while not rospy.is_shutdown():


        arm.exec_position_cmd(np.array([-3.31223050e-04, -1.18001699e+00, -8.22146399e-05, 2.17995802e+00, -2.70787321e-03, 5.69996851e-01,3.32346747e+00]))
        # print arm.exec_position_cmd(np.zeros(7)+0.3)

        # bgr_image = state['rgb_image'][:,:,range(2,-1,-1)]
        # depth_image = state['depth_image']
        # cv2.imshow('captured image', bgr_image)
        #
        # plt.figure(1)

        # # plt.clf()
        # plt.imshow(depth_image, cmap='spectral', interpolation='nearest');
        # plt.figure(2)
        # plt.imshow(state['rgb_image']);
        # plt.draw()
        # plt.show(block=False)
        # plt.pause(0.00001)

        # cv2.waitKey(1)

        # print sawyerArm.get_ee_velocity()

        pb.stepSimulation()

        rate.sleep()

    pb.setRealTimeSimulation(1)

if __name__ == "__main__":    
    main()