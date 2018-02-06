import os
import quaternion
import numpy as np
from aml_io.io_tools import load_data
from os.path import join, dirname, abspath

def load_demo_data(limb_name, demo_idx, debug=False):

    #get the folder that contains the demos
    data_folder_path = '/'.join(dirname(dirname(abspath(__file__))).split('/')[:-2]) + '/data/'

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

            file_path   = data_folder_path + limb_name + '_demo_data' +  ('_sample_%02d.pkl' % demo_idx)
            
            #loads binary file
            demo_data   = load_data(file_path)
        
        except Exception as e:
            print "Data file cannot be loaded, check demo index..."
            raise e

    return demo_data

def get_sampling_rate(limb_name, demo_idx=None, demo_path=None):

    if demo_path is not None:

        demo_data = load_data(demo_path)
    
    else:

        demo_data = load_demo_data(limb_name=limb_name, demo_idx=demo_idx)

    sampling_rate = demo_data[0].get(0,['sampling_rate'])

    if not sampling_rate:

        sampling_rate = 1.

    else:

        sampling_rate = sampling_rate[0]

    return sampling_rate


def get_effort_sequence_of_demo(limb_name, demo_idx):
    
    demo_data   = load_demo_data(limb_name=limb_name, demo_idx=demo_idx)

    jnt_effort_sequence = []

    for arm_data in demo_data['state']:
        
        jnt_effort_sequence.append(arm_data['effort']-arm_data['gravity_comp'])
  
    return np.asarray(jnt_effort_sequence).squeeze()


def js_inverse_dynamics(limb_name, demo_idx, h_component=True):

    demo_data   = load_demo_data(limb_name=limb_name, demo_idx=demo_idx)
    js_pos_traj, js_vel_traj, js_acc_traj = get_js_traj(limb_name=limb_name, demo_idx=demo_idx)
    tau = np.zeros_like(js_acc_traj)
    
    for k in range(len(js_pos_traj)):
    
        if h_component:

            h = demo_data['state'][k]['gravity_comp']

        else:

            h = np.zeros_like(tau[0])

        tau[k] = np.dot(demo_data['state'][k]['inertia'], js_acc_traj[k]) + h

    return tau

def get_os_traj(limb_name, demo_idx, debug=False):

    #NOT COMPATABLE WITH THE NEW DATA RECORDER, CHANGE THIS FUNCTION
    
    demo_data   = load_demo_data(limb_name=limb_name, demo_idx=demo_idx, debug=debug)
    #the limb on which demo was taken
    ee_pos_traj = []
    ee_vel_traj = []
    ee_acc_traj = []
    ee_ori_traj = []
    ee_omg_traj = []
    ee_ang_traj = []


    if debug:

        ee_pos_list = np.asarray(demo_data).squeeze()

    else:

        for arm_data in demo_data['state']:
            ee_pos_traj.append(arm_data['ee_pos'])
            ee_ori_traj.append(arm_data['ee_ori'])
            
            if arm_data.has_key('ee_vel'):
                ee_vel_traj.append(arm_data['ee_vel'])
            
            if arm_data.has_key('ee_omg'):
                ee_omg_traj.append(arm_data['ee_omg'])

        #check is the list is empty
        if ee_vel_traj:
        
            ee_vel_traj =  np.asarray(ee_vel_traj).squeeze()
            ee_acc_traj =  np.diff(ee_vel_traj, axis=0)
            ee_acc_traj =  np.vstack([np.zeros_like(ee_acc_traj[0]), ee_acc_traj])

        #check is the list is empty
        if  ee_omg_traj:

            ee_omg_traj =  np.asarray(ee_omg_traj).squeeze()
            ee_ang_traj =  np.diff(ee_omg_traj, axis=0)
            ee_ang_traj =  np.vstack([np.zeros_like(ee_ang_traj[0]), ee_ang_traj])

        ee_pos_traj =  np.asarray(ee_pos_traj).squeeze()
        ee_ori_traj =  quaternion.as_quat_array(np.asarray(ee_ori_traj).squeeze())

    return ee_pos_traj, ee_ori_traj, ee_vel_traj, ee_omg_traj, ee_acc_traj, ee_ang_traj

def get_js_traj(limb_name, demo_idx=None, demo_path=None):
    #this function takes the position and velocity of a demonstrated data

    if demo_path is not None:
        demo_data = load_data(demo_path)
    else:
        demo_data   = load_demo_data(limb_name=limb_name, demo_idx=demo_idx, debug=False)
    
    sampling_rate = get_sampling_rate(limb_name=limb_name, demo_path=demo_path)
    
    js_pos_traj = []
    js_vel_traj = []
    
    for k in range(demo_data[0].size):
        
        js_pos_traj.append(demo_data[0].get(k,['position']))
        js_vel_traj.append(demo_data[0].get(k,['velocity']))

    js_pos_traj =  np.asarray(js_pos_traj).squeeze()
    js_vel_traj =  np.asarray(js_vel_traj).squeeze()
    #acceleration trajectory using finite differences
    js_acc_traj =  np.diff(js_vel_traj, axis=0)
    js_acc_traj =  np.vstack([np.zeros_like(js_acc_traj[0]), js_acc_traj])*sampling_rate

    print js_pos_traj[0]
    return js_pos_traj, js_vel_traj, js_acc_traj


def plot_demo_data(limb_name, demo_idx):

    ee_list, _, _, _, _, _  = get_os_traj(limb_name=limb_name, demo_idx=demo_idx)

    js_pos_traj, js_vel_traj, js_acc_traj = get_js_traj(limb_name=limb_name, demo_idx=demo_idx)

    #ee_vel_list = np.diff(ee_list, axis=0)
    #print ee_list
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure()
    plt.title('Task space trajectories')
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ee_list[0][0], ee_list[0][1], ee_list[0][2],  linewidths=20, color='r', marker='*')
    ax.scatter(ee_list[-1][0],ee_list[-1][1],ee_list[-1][2], linewidths=20, color='g', marker='*')
    ax.plot(ee_list[:,0],ee_list[:,1],ee_list[:,2])
    #ax.plot(ee_vel_list[:,0],ee_vel_list[:,1],ee_vel_list[:,2], color='m')

    plt.figure(2)
    plt.title('Joint space trajectories')

    num_jnts = js_pos_traj.shape[1]
    plot_no = num_jnts*100 + 11 #just for getting number of joints

    for i in range(num_jnts):
        plt.subplot(plot_no)
        plt.plot(js_pos_traj[:,i])
        plt.plot(js_vel_traj[:,i])
        plt.plot(js_acc_traj[:,i])
        plot_no += 1

    plt.show()



def get_next_possible_traj_index(storage_path, file_name_prefix=None):
    '''
    this function enumerates the number of trajectories in the folder and
    says the next possible index
    '''
    next_index = 1

    if not os.path.exists(storage_path):
        print "The given path to demo does not exist, hence creating a folder"
        os.makedirs(directory)
    else:

        file_list = os.listdir(storage_path)
        file_list.sort() # assuming lexicographic order

        if file_name_prefix is not None:
            for file in file_list:
                 if file.startswith(file_name_prefix):
                    next_index += 1
        else:
            next_index = len(file_list) + 1

    return next_index
