import numpy as np
import random
import rospy
from os.path import dirname, abspath
from aml_robot.mujoco.mujoco_robot import MujocoRobot
from aml_robot.mujoco.mujoco_viewer   import  MujocoViewer

def update_control(robot, cmd=None):

        if cmd is None:

            robot.exec_torque_cmd(np.random.randn(robot._model.nu))

        else:

            robot.exec_torque_cmd(cmd)


def update_magic_forces(robot):

        site_names = robot._model.site_names

        if not site_names:

            print "No sites found to apply inputs"
            raise ValueError


        point1_index = random.randint(0, len(site_names)-1)
        point2_index = random.randint(0, len(site_names)-1)

        point1 = robot._model.site_pose(site_names[point1_index])[0]
        point2 = robot._model.site_pose(site_names[point2_index])[0]
        com    = robot._model.data.xipos[1]

        f_direction1 = (com-point1)/np.linalg.norm(com-point1)
        f_direction2 = (com-point2)/np.linalg.norm(com-point2)

        force1 = 500.*f_direction1# magnitude times direction
        force2 = 500.*f_direction2#

        torque = np.random.randn(3)

        robot._model.data.qfrc_applied = np.hstack([force1+force2, torque])

def main():

    model_folder_path = dirname(dirname(abspath(__file__))) + '/models/'

    # model_name = 'four_two_link_arms_obj.xml'

    # model_name = 'table_setup.xml'

    # model_name = 'four_link_arm.xml'

    model_name = '/baxter/baxter.xml'

    robot      = MujocoRobot(xml_path=model_folder_path+model_name)

    viewer   = MujocoViewer(mujoco_robot=robot)

    robot._configure(viewer=viewer, on_state_callback=True)

    viewer.configure()

    while not rospy.is_shutdown():


        # 

        # update_control(robot)

        # update_magic_forces(robot)

        viewer.loop()


if __name__ == '__main__':

    rospy.init_node('mujoco_tester')

    main()
