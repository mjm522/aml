import os
import numpy as np
import matplotlib.pyplot as plt
from aml_lfd.dmp.discrete_dmp import DiscreteDMP
from aml_lfd.dmp.config import discrete_dmp_config
from aml_lfd.utilities.smooth_demo_traj import SmoothDemoTraj



def train_dmp(trajectory):

    discrete_dmp_config['dof'] = trajectory.shape[1]

    dmp = DiscreteDMP(config=discrete_dmp_config)
    dmp.load_demo_trajectory(trajectory)
    dmp.train()

    return dmp

def test_dmp(dmp):

    discrete_dmp_config['dof'] = dmp._dof

    test_config = discrete_dmp_config
    test_config['dt'] = 0.001

    # play with the parameters
    start_offset = np.zeros(dmp._dof)
    goal_offset = np.zeros(dmp._dof)
    speed = 0.5
    external_force = np.array([0.,0.,0.,0.])
    alpha_phaseStop = 20.

    test_config['y0'] = dmp._traj_data[0, 1:] + start_offset
    test_config['dy'] = np.zeros(dmp._dof) 
    test_config['goals'] = dmp._traj_data[-1, 1:] + goal_offset
    test_config['tau'] = 1./speed
    test_config['ac'] = alpha_phaseStop
    test_config['type'] = 1

    if test_config['type'] == 3:
        test_config['extForce'] = external_force
    else:
        test_config['extForce'] = np.array([0,0,0,0])

    gen_traj = dmp.generate_trajectory(config=test_config)

    # tmp = dmp.generate_trajectory_old(test_config)

    time_stamps = gen_traj['time_stamps']
    test_traj   = gen_traj['pos']

    #in 2D only maximum only two dimensions can be plotted

    if dmp._dof > 2:
        print "Warning: Only 2 dimensions can be plotted"

    plt.figure("dmp-pos x vs y")
    plt.plot(dmp._traj_data[:,1], dmp._traj_data[:,2], 'b-')
    plt.plot(test_traj[:,0], test_traj[:,1], 'r--')

    # plt.figure("OLD dmp-pos x vs y")
    # plt.plot(dmp._traj_data[:,1], dmp._traj_data[:,2], 'b-')
    # plt.plot(tmp['pos'][:,0], tmp['pos'][:,1], 'r--')

    plt.figure("dmp-pos vs time")
    for k in range(dmp._dof):

        plt.plot(time_stamps, test_traj[:,k])

    # plt.figure("OLD dmp-pos vs time")
    # for k in range(dmp._dof):

    #     plt.plot(time_stamps, tmp['pos'][:,k])

    plt.show()

def main():

    data_storage_path = os.environ['AML_DATA'] + '/aml_lfd/dmp/'
    file_name = data_storage_path+'recorded_trajectory.txt'

    trajectory = np.loadtxt(file_name)

    #this is for smoothing the trajectory
    smooth_traj = SmoothDemoTraj(trajectory, window_len=35, poly_order=3)

    dmp = train_dmp(smooth_traj._smoothed_traj)
    test_dmp(dmp)


if __name__ == '__main__':
    main()