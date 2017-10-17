import os
import rospy
import numpy as np
import matplotlib.pyplot as plt
from aml_lfd.dmp.config import discrete_dmp_config
from aml_lfd.dmp.discrete_dmp_shell import DiscreteDMPShell


def extract_js_traj(filename, save_name):
    trajectory = np.load(file_name)
    demo_js_traj = []
    for data in trajectory:
        demo_js_traj.append(data['position'])

    demo_js_traj = np.asarray(demo_js_traj)
    demo_js_traj = np.savetxt(save_name, demo_js_traj)

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

def test_dmp(dmp):
    test_config = discrete_dmp_config
    test_config['dt'] = 0.001

    # play with the parameters
    start_offset = np.zeros(discrete_dmp_config['dof'])
    goal_offset = np.zeros(discrete_dmp_config['dof'])
    speed = 1.
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

    plot_traj([dmp._traj_data[:,1:], test_traj[:,1:]])

def main():

    data_storage_path = os.environ['AML_DATA'] + '/aml_lfd/'
    file_name = data_storage_path+'demo_data_6.npy'

    # extract_js_traj(filename=filename, save_name="../data/demo_traj_6.txt")

    trajectory = np.loadtxt("../data/demo_traj_6.txt")
    # plot_traj([trajectory])
    dmp = train_dmp(trajectory)
    test_dmp(dmp)


if __name__ == '__main__':
    main()

