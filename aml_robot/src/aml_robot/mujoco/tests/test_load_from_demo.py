import numpy as np
import random
import rospy
import copy
from os.path import dirname, abspath
from aml_lfd.utilities.utilities import get_effort_sequence_of_demo, get_js_traj, get_sampling_rate
from aml_robot.mujoco.mujoco_robot import MujocoRobot
from aml_robot.mujoco.mujoco_viewer   import  MujocoViewer

def update_control(robot, cmd_r, cmd_l):

    # robot.exec_position_cmd(combined_cmd)

    if cmd_r is None:

        cmd_r = np.zeros((7,1))

    if cmd_l is None:

        cmd_l = np.zeros((7,1))

    # curr_qpos = copy.deepcopy(robot._model.data.qpos)

    # curr_qpos = curr_qpos.squeeze()

    # curr_qpos[1:8]   = cmd_r
    # curr_qpos[10:17] = cmd_l

    combined_cmd = np.vstack([0., cmd_r, np.zeros((2,1)), cmd_l, np.zeros((2,1))])

    # robot.exec_position_cmd(combined_cmd)

    # robot.move_to_joint_pos(curr_qpos)
    robot.exec_torque_cmd(combined_cmd)

def compute_inverse_dyn(model_folder_path, model_name, jnt_pos, jnt_vel, jnt_acc):

    #create a local robot object
    #apply the positions, vels and accelerations.
    robot         = MujocoRobot(xml_path=model_folder_path+model_name)

    combined_qpos = copy.deepcopy(robot._model.data.qpos)
    combined_qvel = copy.deepcopy(robot._model.data.qvel)
    combined_qacc = copy.deepcopy(robot._model.data.qacc)

    combined_qpos[10:17] = jnt_pos.reshape(7,1)
    combined_qvel[10:17] = jnt_vel.reshape(7,1)
    combined_qacc[10:17] = jnt_acc.reshape(7,1)

    inv_dyn = robot.inv_dyn(combined_qpos, combined_qvel, combined_qacc)

    return inv_dyn[10:17]


def main():

    model_folder_path = dirname(dirname(abspath(__file__))) + '/models/'

    model_name = '/baxter/baxter.xml'

    robot      = MujocoRobot(xml_path=model_folder_path+model_name)

    viewer     = MujocoViewer(mujoco_robot=robot)

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

    torque_cmds = []
    not_saved = True

    limb_masses = np.array([5.7004, 3.2270, 4.312, 2.0721, 2.2466, 1.6098, 0.3509+0.1913])

    inv_limb_masses = np.array([0.17542629, 0.30988534, 0.23191095, 0.48260219, 0.44511707, 0.62119518, 1.84433788])

    normalized_gaind = np.array([0.28921213, 0.16372317, 0.21877109, 0.10512884, 0.11398217, 0.08167386, 0.02750874])

    curr_qpos = copy.deepcopy(robot._model.data.qpos)
    curr_qpos[1:8]   = untuck_r.reshape(7,1) 
    curr_qpos[10:17] = untuck_l.reshape(7,1) 
    
    robot.set_qpos(curr_qpos)

    while not rospy.is_shutdown():
     
        if not completed:

            for k in range(len(js_pos_traj)):

                # print "k \t", k

                torque = compute_inverse_dyn(model_folder_path, 
                                             model_name, 
                                             js_pos_traj[k], 
                                             js_vel_traj[k], 
                                             js_acc_traj[k])

                # print "torque \t", np.round(torque.reshape(1,7), 3)

                update_control(robot, None, torque)
        
                viewer.loop()

                # rate.sleep()
            update_control(robot, None, None)
            print "Done"
            completed = True

        viewer.loop()


if __name__ == '__main__':

    rospy.init_node('mujoco_tester')

    main()
