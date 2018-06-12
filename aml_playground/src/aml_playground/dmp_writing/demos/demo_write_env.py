#!/usr/bin/env python

import rospy
import numpy as np
from matplotlib import pyplot as plt
from aml_playground.dmp_writing.write_worlds.write_world_env import WriteEnv

def main():

    rospy.init_node('write_env', anonymous=True)

    write_env = WriteEnv()

    rate = rospy.Rate(500)

    arm = write_env._sawyer

    arm._bullet_robot.set_ctrl_mode('pos')

    while not rospy.is_shutdown():

        arm.exec_position_cmd(np.array([-3.31223050e-04, -1.18001699e+00, -8.22146399e-05, 2.17995802e+00, -2.70787321e-03, 5.69996851e-01,3.32346747e+00]))

        write_env.step()

        rate.sleep()

if __name__ == "__main__":    
    main()