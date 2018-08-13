import os
import copy
import pandas
import numpy as np
import quaternion as q
import scipy.signal as sig
import matplotlib.pyplot as plt
from aml_io.io_tools import load_data
from scipy.interpolate import interp1d



data_path = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/sinusoid_exps/'

file_list = ['with_k1.pkl',
             'with_k1_5.pkl',
             'with_k2.pkl',
             'with_k2_5.pkl',
             'with_k3.pkl',
             'with_k3_5.pkl',
             'with_k4.pkl',
             'with_k4_5.pkl',
             'with_k5.pkl']

def lpf(data):
    b, a = sig.butter(3, 0.05)
    y = sig.filtfilt(b, a, data)
    return y


def plot_params(data):

    ee_traj = data['ee_traj']
    ee_vel_traj = data['ee_vel_traj']
    ee_wrenches_local = data['ee_wrenches_local']
    req_traj = [data['other_ee_data'][i]['req_traj'] for i in range(len(ee_traj))]
    pos_traj = [data['other_ee_data'][i]['position'] for i in range(len(ee_traj))]
    ee_wrenches = data['ee_wrenches']
    plt.figure("Spring Pulling - Different Stiffness")
    
    plt.subplot(3,5,1)
    plt.title("q_1")
    plt.plot(np.asarray(pos_traj)[:,0], 'g')
    plt.subplot(3,5,2)
    plt.title("Px")
    plt.plot(ee_traj[:,0], 'g')
    plt.plot(np.asarray(req_traj)[:,0], 'r')
    plt.subplot(3,5,3)
    plt.title("Fx")
    plt.plot(ee_wrenches[:,0])
    plt.subplot(3,5,4)
    plt.title("Fx_local")
    plt.plot(ee_wrenches_local[:,0])
    plt.subplot(3,5,5)
    plt.title("Vx")
    plt.plot(ee_vel_traj[:,0])

    
    plt.subplot(3,5,6)
    plt.title("q_2")
    plt.plot(np.asarray(pos_traj)[:,1], 'g')
    plt.subplot(3,5,7)
    plt.title("Py")
    plt.plot(ee_traj[:,1], 'g')
    plt.plot(np.asarray(req_traj)[:,1], 'r')
    plt.subplot(3,5,8)
    plt.title("Fy")
    plt.plot(ee_wrenches[:,1])
    plt.subplot(3,5,9)
    plt.title("Fy_local")
    plt.plot(ee_wrenches_local[:,1])
    plt.subplot(3,5,10)
    plt.title("Vy")
    plt.plot(ee_vel_traj[:,1])

    
    plt.subplot(3,5,11)
    plt.title("q_3")
    plt.plot(np.asarray(pos_traj)[:,2], 'g')
    plt.subplot(3,5,12)
    plt.title("Pz")
    plt.plot(ee_traj[:,2], 'g')
    plt.plot(np.asarray(req_traj)[:,2], 'r')
    plt.subplot(3,5,13)
    plt.title("Fz")
    plt.plot(ee_wrenches[:,2])
    plt.subplot(3,5,14)
    plt.title("Fz_local")
    plt.plot(ee_wrenches_local[:,2])
    plt.subplot(3,5,15)
    plt.title("Vz")
    plt.plot(ee_vel_traj[:,2])
    # plt.show()


def visualise_data(show_plots = True): #right2left left2right


    #contacts_data_left2right_changing_angles
    # file = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/contacts_data_'+file_+'_changing_angles.pkl'

    pos_list, ori_list, lin_vels_list, ang_vels_list, f_list, t_list, pos_list = [],[],[],[],[],[],[]

    plt.ion()
    for data_file in file_list:

        file = data_path + data_file
        data = load_data(file)

        
        plot_params(data)



    #     pos_list.append(data['ee_traj'])
    #     f_list.append(data['ee_wrenches_local'])
    #     # ori_list.append([data['tip_state']['orientation'] for i in range(N)])
    #     lin_vels_list.append(['ee_vel_traj'])
    #     # ang_vels_list.append([data['tip_state']['angular_vel'] for i in range(N)])

    #     # ft_reading = convert_list_str_ft_reading([data['ft_reading'] for i in range(N)])
    #     # f_list.append(ft_reading[:, :3])
    #     # t_list.append(ft_reading[:, 3:])

    # for k in range(len(file_list)):

    #     file_name   = str(file_list[k])

    #     fig_force = plot_(left_data=np.asarray(pos_list[k]),
    #                   right_data=np.asarray(f_list[k]),
    #                   f_title="force - extension",
    #                   p_title=file_name,
    #                   names=['Px', 'Fx', 'Py', 'Fy', 'Pz', 'Fz'],
    #                   do_clf=False,
    #                   ylabels=['N', 'N'])

    #     plt.draw()
        # plt.pause(0.00001)
        raw_input()

if __name__ == '__main__':


    visualise_data()
