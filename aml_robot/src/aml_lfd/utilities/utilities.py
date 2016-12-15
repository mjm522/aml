import numpy as np
import os 

def load_demo_data(demo_idx=1, debug=False):

    #get the parent folder
    data_folder_path = os.path.normpath(os.getcwd() + os.sep + os.pardir) + '/data/'

    if debug:
        #for debugging purposes, load a known trajectory
        try:
            file_path = data_folder_path + 'debug.txt'
            demo_data = np.loadtxt(file_path, dtype='float', delimiter=',')

        except Exception as e:
            print "Data file cannot be loaded"
            raise e

    else:

        try:

            #add the file name of the stored demo file
            file_path   = data_folder_path + 'demo_data_' + str(demo_idx) + '.npy'
            #loads binary file
            demo_data   = np.load(file_path)
        
        except Exception as e:
            print "Data file cannot be loaded, check demo index..."
            raise e

    return demo_data

def get_ee_traj(demo_idx=1, debug=False):
    
    demo_data = load_demo_data(demo_idx=demo_idx, debug=debug)
    ee_list = []

    if debug:

        ee_list = np.asarray(demo_data).squeeze()

    else:

        for arm_data in demo_data:
            ee_list.append(arm_data['ee_pos'])

        ee_list =  np.asarray(ee_list).squeeze()

    return ee_list

def plot_demo_data(demo_idx=1):

    ee_list = get_ee_traj(demo_idx=demo_idx)
    #ee_vel_list = np.diff(ee_list, axis=0)
    #print ee_list
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ee_list[0][0], ee_list[0][1], ee_list[0][2],  linewidths=20, color='r', marker='*')
    ax.scatter(ee_list[-1][0],ee_list[-1][1],ee_list[-1][2], linewidths=20, color='g', marker='*')
    ax.plot(ee_list[:,0],ee_list[:,1],ee_list[:,2])
    #ax.plot(ee_vel_list[:,0],ee_vel_list[:,1],ee_vel_list[:,2], color='m')
    plt.show()