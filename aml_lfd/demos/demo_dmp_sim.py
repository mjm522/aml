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


def update_dmp_params(dmp, start_offset, goal_offset, speed):
    
    discrete_dmp_config['dof'] = dmp._dof

    test_config = discrete_dmp_config
    test_config['dt'] = 0.001

    # play with the parameters
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

    return gen_traj


def test_dmp(dmp):

    start_offset1 = np.zeros(dmp._dof)
    goal_offset1 = np.zeros(dmp._dof)

    gen_traj_speed1 = update_dmp_params(dmp, np.zeros(dmp._dof), np.zeros(dmp._dof), 1.5)
    gen_traj_speed2 = update_dmp_params(dmp, np.zeros(dmp._dof), np.zeros(dmp._dof), 1.)
    gen_traj_speed3 = update_dmp_params(dmp, np.zeros(dmp._dof), np.zeros(dmp._dof), 0.5)

    gen_traj_spatial1 = update_dmp_params(dmp, np.zeros(dmp._dof), np.array([1.,1.]), 1.)
    gen_traj_spatial2 = update_dmp_params(dmp, np.zeros(dmp._dof), np.zeros(dmp._dof), 1.)
    gen_traj_spatial3 = update_dmp_params(dmp, np.zeros(dmp._dof), np.array([-1.,-1.]), 1.)

    #in 2D only maximum only two dimensions can be plotted

    if dmp._dof > 2:
        print "Warning: Only 2 dimensions can be plotted"

    plt.figure("dmp-pos x vs y - tempoal")
    plt.plot(dmp._traj_data[:,1], dmp._traj_data[:,2], 'b-')
    plt.plot(gen_traj_speed1['pos'][:,0], gen_traj_speed1['pos'][:,1], 'r--')
    plt.plot(gen_traj_speed2['pos'][:,0], gen_traj_speed2['pos'][:,1], 'g--')
    plt.plot(gen_traj_speed3['pos'][:,0], gen_traj_speed3['pos'][:,1], 'b--')
    plt.title("DMP-Demo Traj - Temporal Scale")
    plt.xlabel("distance X (m)")
    plt.ylabel("distance Y (m)")

    plt.figure("dmp-pos x vs y - spatial")
    plt.plot(dmp._traj_data[:,1], dmp._traj_data[:,2], 'b-')
    plt.plot(gen_traj_spatial1['pos'][:,0], gen_traj_spatial1['pos'][:,1], 'r--')
    plt.plot(gen_traj_spatial2['pos'][:,0], gen_traj_spatial2['pos'][:,1], 'g--')
    plt.plot(gen_traj_spatial3['pos'][:,0], gen_traj_spatial3['pos'][:,1], 'b--')
    plt.title("DMP-Demo Traj - Spatial Scale")
    plt.xlabel("distance X (m)")
    plt.ylabel("distance Y (m)")

    plt.figure("dmp-pos vs time - spatial")
    labels= ['x+1; tau=1', 'y+1; tau=1']
    for k in range(dmp._dof):
        plt.plot(gen_traj_spatial1['time_stamps'], gen_traj_spatial1['pos'][:,k], label=labels[k])

    labels= ['x+0; tau=1', 'y+0; tau=1']
    for k in range(dmp._dof):
        plt.plot(gen_traj_spatial2['time_stamps'], gen_traj_spatial2['pos'][:,k], label=labels[k])

    labels= ['x-1; tau=1', 'y-1; tau=1']
    for k in range(dmp._dof):
        plt.plot(gen_traj_spatial3['time_stamps'], gen_traj_spatial3['pos'][:,k], label=labels[k])

    plt.xlabel("time (s)")
    plt.ylabel("distance (m)")
    plt.title("DMP- X and Y Traj - Spatial")
    plt.legend()

    plt.figure("dmp-pos vs time - temporal")
    labels= ['x+0; tau=1.5', 'y+0+0; tau=1.5']
    for k in range(dmp._dof):
        plt.plot(gen_traj_speed1['time_stamps'], gen_traj_speed1['pos'][:,k], label=labels[k])

    labels= ['x+0; tau=1.', 'y+0; tau=1.']
    for k in range(dmp._dof):
        plt.plot(gen_traj_speed2['time_stamps'], gen_traj_speed2['pos'][:,k], label=labels[k])

    labels= ['x+0; tau=0.5', 'y+0; tau=0.5']
    for k in range(dmp._dof):
        plt.plot(gen_traj_speed3['time_stamps'], gen_traj_speed3['pos'][:,k], label=labels[k])

    plt.xlabel("time (s)")
    plt.ylabel("distance (m)")
    plt.title("DMP- X and Y Traj - Temporal")
    plt.legend()

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