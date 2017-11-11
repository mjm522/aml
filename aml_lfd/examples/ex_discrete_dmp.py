import os
import rospy
import numpy as np
import matplotlib.pyplot as plt
from aml_lfd.dmp.config import discrete_dmp_config
from aml_ctrl.traj_player.traj_player import TrajPlayer
from aml_lfd.dmp.discrete_dmp_shell import DiscreteDMPShell
from aml_ctrl.traj_generator.js_traj_generator import JSTrajGenerator
from aml_ctrl.controllers.js_controllers.js_postn_controller import JSPositionController


def extract_js_traj(filename, limb_name):
    kwargs = {}
    kwargs['path_to_demo'] = filename
    kwargs['limb_name'] = limb_name
    gen_traj    = JSTrajGenerator(load_from_demo=True, **kwargs)
    trajectory = gen_traj.generate_traj()
    return trajectory

def plot_traj(trajectories):
    plt.figure("trajectories")
    
    for i in range(len(trajectories)):
        trajectory = trajectories[i]
        idx = trajectory.shape[1]*100 + 11
        if i == 0:
            color = 'r--'
        elif i==1:
            color = 'g'
        else:
            color = 'b'
        for k in range(trajectory.shape[1]):
            plt.subplot(idx)
            idx += 1
            plt.plot(trajectory[:,k], color)
    plt.show() 


def train_dmp(trajectory):
    discrete_dmp_config['dof'] = 7
    dmp = DiscreteDMPShell(config=discrete_dmp_config)
    dmp.load_demo_trajectory(trajectory)
    dmp.train()

    return dmp

def test_dmp(dmp, speed=1., plot_trained=False):
    test_config = discrete_dmp_config
    test_config['dt'] = 0.001

    # play with the parameters
    start_offset = np.zeros(discrete_dmp_config['dof'])
    goal_offset = np.zeros(discrete_dmp_config['dof'])
    external_force = np.zeros(discrete_dmp_config['dof'])
    alpha_phaseStop = 20.

    test_config['y0'] = dmp._traj_data[0, 1:] + start_offset
    test_config['dy'] = np.zeros(discrete_dmp_config['dof'])
    test_config['goals'] = dmp._traj_data[-1, 1:] + goal_offset
    test_config['tau'] = 1./speed
    test_config['ac'] = alpha_phaseStop
    test_config['type'] = 1
    test_config['dof'] = 7

    if test_config['type'] == 3:
        test_config['extForce'] = external_force
    else:
        test_config['extForce'] = np.zeros(discrete_dmp_config['dof'])
    test_traj = dmp.test(config=test_config)

    if plot_trained:
        plot_traj([dmp._traj_data[:,1:], test_traj[:,1:]])


    vel_traj =  np.diff(test_traj[:,1:], axis=0)
    vel_traj =  np.vstack([np.zeros_like(vel_traj[0]), vel_traj])*test_config['dt']
    acc_traj =  np.diff(vel_traj, axis=0)
    acc_traj =  np.vstack([np.zeros_like(acc_traj[0]), acc_traj])*test_config['dt']

    test_traj = {
    'pos_traj': test_traj[:,1:],
    'vel_traj':vel_traj,
    'acc_traj':acc_traj
    }
    
    return test_traj


def setup_robot():
    from aml_robot.baxter_robot import BaxterArm
    limb = 'right'
    arm = BaxterArm(limb)
    arm.untuck_arm()

    return arm


def test_on_robot(robot_interface, des_path):

    traj_player = TrajPlayer(robot_interface=robot_interface, 
                             controller=JSPositionController, 
                             trajectory=des_path, 
                             rate=100)
    traj_player.player()

def main():

    rospy.init_node('dmp_demo')
    
    limb = 'left'
    path_to_demo = os.environ['AML_DATA'] + '/aml_lfd/dmp/' + limb + '/' + limb +'_dmp_01.pkl'

    from aml_robot.baxter_robot import BaxterArm

    arm = BaxterArm(limb)

    if not os.path.isfile(path_to_demo):
        raise Exception("The given path to demo does not exist, given path: \n" + path_to_demo)

    trajectory = extract_js_traj(filename=path_to_demo, limb_name=limb)

    dmp = train_dmp(trajectory['pos_traj'])
    test_traj = test_dmp(dmp, speed=5.,plot_trained=False)
    
    test_on_robot(robot_interface=arm,
                  des_path=test_traj)


if __name__ == '__main__':
    main()

