import numpy as np
import random
import rospy
from os.path import dirname, abspath
from aml_lfd.utilities.utilities import get_effort_sequence_of_demo, get_js_traj, get_sampling_rate
from aml_robot.mujoco.mujoco_robot import MujocoRobot
from aml_robot.mujoco.mujoco_viewer   import  MujocoViewer

def update_control(robot, cmd_r, cmd_l):

    combined_cmd = np.hstack([0., cmd_r, np.zeros(2), cmd_l, np.zeros(2)])

    robot.exec_position_cmd(combined_cmd)
    # robot.exec_torque_cmd(combined_cmd)


def main():

    model_folder_path = dirname(dirname(abspath(__file__))) + '/models/'

    model_name = '/baxter/baxter.xml'

    robot      = MujocoRobot(xml_path=model_folder_path+model_name)

    viewer   = MujocoViewer(mujoco_robot=robot)

    robot._configure(viewer=viewer, on_state_callback=True)

    viewer.configure()

    effort_sequence = get_effort_sequence_of_demo('left', 0)
    
    js_pos_traj, js_vel_traj, js_acc_traj = get_js_traj('left', 0)

    sampling_rate = get_sampling_rate('left', 0)

    rate = rospy.Rate(sampling_rate)

    completed = False

    tuck   = np.array([-1.0,  -2.07,  3.0,  2.55,  0.0,  0.01,  0.0])

    untuck_l = np.array([-0.08, -1.0,  -1.19, 1.94,  0.67, 1.03, -0.50])

    untuck_r = np.array([0.08, -1.0,   1.19, 1.94,  -0.67, 1.03,  0.50])

    while not rospy.is_shutdown():

        if not completed:

            # for effort in effort_sequence:

            update_control(robot, untuck_r, untuck_l)

            # for jnt_pos in js_pos_traj:

            #     # combined_cmd =  robot.get_compensation_forces() #+ np.hstack([np.zeros(7), effort]) 

            #     # print "effort \t", np.round(effort,3)

            #     # print robot._nu

            #     update_control(robot, jnt_pos)

            #     viewer.viewer_render()

            #     rate.sleep()

            completed = True

        viewer.viewer_render()


if __name__ == '__main__':

    rospy.init_node('mujoco_tester')

    main()
