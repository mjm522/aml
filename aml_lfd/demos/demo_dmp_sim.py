import os
import numpy as np
import matplotlib.pyplot as plt
from aml_lfd.dmp.config import discrete_dmp_config
from aml_lfd.dmp.discrete_dmp_shell import DiscreteDMPShell


def train_dmp(trajectory):

    dmp = DiscreteDMPShell(config=discrete_dmp_config)
    dmp.load_demo_trajectory(trajectory)
    dmp.train()

    return dmp

def test_dmp(dmp):

    test_config = discrete_dmp_config
    test_config['dt'] = 0.001

    # play with the parameters
    start_offset = np.array([0.,0.])
    goal_offset = np.array([.0, 0.])
    speed = 0.5
    external_force = np.array([0.,0.,0.,0.])
    alpha_phaseStop = 20.

    test_config['y0'] = dmp._traj_data[0, 1:] + start_offset
    test_config['dy'] = np.array([0., 0.])
    test_config['goals'] = dmp._traj_data[-1, 1:] + goal_offset
    test_config['tau'] = 1./speed
    test_config['ac'] = alpha_phaseStop
    test_config['type'] = 1

    if test_config['type'] == 3:
        test_config['extForce'] = external_force
    else:
        test_config['extForce'] = np.array([0,0,0,0])

    test_traj = dmp.test(config=test_config)

    plt.figure(1)
    plt.plot(dmp._traj_data[:,1], dmp._traj_data[:,2], 'b-')
    plt.plot(test_traj[:,1], test_traj[:,2], 'r--')

    plt.figure(2)
    plt.plot(test_traj[:,0], test_traj[:,1], 'g-')
    plt.plot(test_traj[:,0], test_traj[:,2], 'm-')
    plt.show()

def main():

    data_storage_path = os.environ['AML_DATA'] + '/aml_lfd/dmp/'
    file_name = data_storage_path+'recorded_trajectory.txt'
    trajectory = np.loadtxt(file_name)

    dmp = train_dmp(trajectory)
    test_dmp(dmp)


if __name__ == '__main__':
    main()