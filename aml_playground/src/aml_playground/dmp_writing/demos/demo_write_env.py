#!/usr/bin/env python

import os
import rospy
import numpy as np
import quaternion
import csv
from matplotlib import pyplot as plt
from aml_rl_envs.utils.data_utils import load_csv_data, save_csv_data
from aml_playground.dmp_writing.write_worlds.write_world_env import WriteEnv
from aml_rl_envs.utils.collect_demo import plot_demo

def main():

    rospy.init_node('write_env', anonymous=True)

    write_env = WriteEnv()

    rate = rospy.Rate(500)

    arm = write_env._sawyer

    arm._bullet_robot.set_ctrl_mode('pos')

    filepath = os.environ['AML_DATA'] + '/dmp_writing/infinity.csv'

    filepath_ts = os.environ['AML_DATA'] + '/dmp_writing/infinity_ts_ee.csv' #task_space end effector values
    filepath_ts_o = os.environ['AML_DATA'] + '/dmp_writing/infinity_ts_ori.csv' #task_space orientation values

    # csvfile = open(filepath)
    # reader = csv.reader(csvfile)

    file_contents = load_csv_data(filepath) #trajectory values required to plot

    # plot_demo(trajectory = file_contents, color=[0,0,1], start_idx=0, life_time=0., cid=write_env._cid) # plot function to plot the trajectories

    ee_poses = [] #initialisation of the end effector pose and orientation array
    ee_oris = []
            
    while not rospy.is_shutdown():
            
        for row in file_contents:

            # cmds = arm.inverse_kinematics(row, (0.,1.,0.,0.)) #convertion of joint_space to task_space

            arm.exec_position_cmd(row)

            write_env.step()

            ee_poses.append(arm.ee_pose()[0])
            ee_oris.append(quaternion.as_float_array(arm.ee_pose()[1]))

            rate.sleep()

        save_csv_data(filepath_ts, ee_poses)
        save_csv_data(filepath_ts_o, ee_oris)
        break

if __name__ == "__main__":    
    main()