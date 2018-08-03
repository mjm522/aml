import numpy as np
import matplotlib.pyplot as plt

def plot_params(data):

    w_list = np.asarray(data['params'])
    spring_force = data['spring_force']
    force_traj = data['ee_wrenches_local'][:,:3]
    req_traj = data['traj']
    pos_traj = data['ee_traj']
    vel_traj = data['ee_vel_traj']
    plt.figure("Varying stiffness plot")
    
    plt.subplot(3,5,1)
    plt.title("Kx")
    plt.plot(w_list[:,0])
    plt.subplot(3,5,2)
    plt.title("Px")
    plt.plot(pos_traj[:,0], 'g')
    plt.plot(req_traj[:,0], 'r')
    plt.subplot(3,5,3)
    plt.title("Vx")
    plt.plot(vel_traj[:,0])
    plt.subplot(3,5,4)
    plt.title("Fx")
    plt.plot(force_traj[:,0])
    plt.subplot(3,5,5)
    plt.title("Sx")
    plt.plot(spring_force[:,0])

    
    plt.subplot(3,5,6)
    plt.title("Ky")
    plt.plot(w_list[:,1])
    plt.subplot(3,5,7)
    plt.title("Py")
    plt.plot(pos_traj[:,1], 'g')
    plt.plot(req_traj[:,1], 'r')
    plt.subplot(3,5,8)
    plt.title("Vy")
    plt.plot(vel_traj[:,1])
    plt.subplot(3,5,9)
    plt.title("Fy")
    plt.plot(force_traj[:,1])
    plt.subplot(3,5,10)
    plt.title("Sy")
    plt.plot(spring_force[:,1])

    
    plt.subplot(3,5,11)
    plt.title("Kz")
    plt.plot(w_list[:,2])
    plt.subplot(3,5,12)
    plt.title("Pz")
    plt.plot(pos_traj[:,2], 'g')
    plt.plot(req_traj[:,2], 'r')
    plt.subplot(3,5,13)
    plt.title("Vz")
    plt.plot(vel_traj[:,2])
    plt.subplot(3,5,14)
    plt.title("Fz")
    plt.plot(force_traj[:,2])
    plt.subplot(3,5,15)
    plt.title("Sz")
    plt.plot(spring_force[:,2])
    plt.show()

    plt.plot(force_traj[:,2], w_list[:,2])
    
    plt.show()