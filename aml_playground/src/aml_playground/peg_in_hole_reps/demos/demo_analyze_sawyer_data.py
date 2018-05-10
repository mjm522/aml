import os
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from aml_io.io_tools import load_data

demo_data_path = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/sawyer/demo_data/'

demo_data_files = ['right_sawyer_exp_imp_learn_90_degree.pkl',
                    'right_sawyer_exp_imp_learn_side_1_bkwd.pkl',
                    # 'right_sawyer_exp_imp_learn_side_2_bkwd.pkl',
                    # 'right_sawyer_exp_imp_learn_side_3_bkwd.pkl',
                    # 'right_sawyer_exp_imp_learn_side_4_bkwd.pkl',
                    'right_sawyer_exp_imp_learn_side_1_fwd.pkl',]
                    # 'right_sawyer_exp_imp_learn_side_2_fwd.pkl',
                    # 'right_sawyer_exp_imp_learn_side_3_fwd.pkl',
                    # 'right_sawyer_exp_imp_learn_side_4_fwd.pkl']

replay_data_path = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/sawyer/replayed_data/data_with_ft/'

replay_data_files = ['sawyer_replayed_data_90_degree.pkl',
                    'sawyer_replayed_data_side_1_bkwd.pkl',
                     'sawyer_replayed_data_side_2_bkwd.pkl',
                     'sawyer_replayed_data_side_3_bkwd.pkl',
                     'sawyer_replayed_data_side_4_bkwd.pkl',
                     'sawyer_replayed_data_side_1_fwd.pkl',
                     'sawyer_replayed_data_side_2_fwd.pkl',
                     'sawyer_replayed_data_side_3_fwd.pkl',
                     'sawyer_replayed_data_side_4_fwd.pkl']

# data['tip_state']['linear_vel']
# data['tip_state']['angular_vel']
# data['tip_state']['force']
# data['tip_state']['torque']

def lpf(data):
    b, a = sig.butter(3, 0.05)
    y = sig.filtfilt(b, a, data)
    return y


def visualise_data(file_='right2left'): #right2left left2right

    file_list = replay_data_files
    data_path = replay_data_path#replay_data_path

    #contacts_data_left2right_changing_angles
    # file = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/contacts_data_'+file_+'_changing_angles.pkl'

    pos_list, ori_list, lin_vels_list, ang_vels_list, f_list, t_list, pos_list = [],[],[],[],[]

    for data_file in file_list:

        file = data_path + data_file
        data = load_data(file)[0].get_contents()

        pos_list.append([data[i]['tip_state']['position'] for i in range(len(data))])
        ori_list.append([data[i]['tip_state']['orientation'] for i in range(len(data))])
        lin_vels_list.append([data[i]['tip_state']['linear_vel'] for i in range(len(data))])
        ang_vels_list.append([data[i]['tip_state']['angular_vel'] for i in range(len(data))])
        # f_list.append([data[i]['tip_state']['force'] for i in range(len(data))])
        # t_list.append([data[i]['tip_state']['torque'] for i in range(len(data))])

        ft_reading = []
        for i in range(len(data)):
            strft = data[i]['ft_reading'].split(' ')
            ft_reading_array = np.zeros(6)
            k = 0
            for j in range(len(strft)):
                try:
                    val = float(strft[j])
                except Exception as e:
                    continue
                ft_reading_array[k] = val
                k += 1
                if k > 5:
                    break
            ft_reading.append(ft_reading_array)

        ft_reading = np.asarray(ft_reading)
        f_list.append(ft_reading[:, :3])
        t_list.append(ft_reading[:, 3:])

        # ft_reading = [data[i]['ft_reading'].split(' ')[j] for i in range(len(data)) if data[i]['ft_reading'].split(' ')[i] != '']
        # print ft_reading
        # for i in range(len(data)):

        #     ft_reading = data[i]['ft_reading'].split(' ')
        #     print ft_reading
        #     f_reading = np.array([ float(ft_reading[0]), float(ft_reading[2]), float(ft_reading[4]) ])
        #     t_reading = np.array([ float(ft_reading[5]), float(ft_reading[7]), float(ft_reading[9])])
        #     f_list.append(f_reading)
        #     t_list.append(t_reading)

        pos_list.append([data[i]['tip_state']['position'] for i in range(len(data))])


    # print data[0].get_contents()
    # print "done"
    # raw_input()

    # data_to_show = [0,5,10,15,20]

    # col = ['r','g','b','y','c']
    # plt.figure("force")
    # plt.figure("traj")
    plt.ion()

    for k in range(len(file_list)):

        # ee_traj  = i['ee_traj']
        # ee_wrenches = i['ee_wrenches']
        # ee_wrenches_local = i['ee_wrenches_local']  

        print k+1, file_list[k]

        forces_list  = np.asarray(f_list[k]) #ee_wrenches[:,:3]
        torques_list = np.asarray(t_list[k]) #ee_wrenches[:,3:]
        pos = np.asarray(pos_list[k]) #ee_wrenches[:,3:]
        l_vel = np.asarray(lin_vels_list[k])
        a_vel = np.asarray(ang_vels_list[k])
        # forces_list_local  = ee_wrenches_local[:,:3]
        # torques_list_local = ee_wrenches_local[:,3:]

        l_idx = 0
        g_idx = 0

        file_ = str(('_'.join(file_list[k].split('_')[3:])).split('.')[0])

        file_name_force = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/' + file_+ '_images_force_' + str(k)+ '.png'
        file_name_traj = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/' + file_ +'_images_traj_' + str(k)+ '.png'

        subplot_idx = 320   
        # color = col[k]
        plt.figure("force - torque", figsize=(15, 15))
        plt.title(file_)
        # plt.clf()
        names = ['Fx', 'Tx', 'Fy', 'Ty', 'Fz', 'Tz']
        for j in range(6):
            
            subplot_idx += 1
            ax = plt.subplot(subplot_idx)
            ax.set_title(names[j])
            plt.xlabel("num data")
            if j%2 == 1:
                ax.plot(lpf(torques_list[:,l_idx]), label=file_)
                l_idx += 1
                plt.ylabel("Nm")
                # if j == 1:
                #     plt.legend(replay_data_files)
            else:
                plt.plot(lpf(forces_list[:,g_idx]), label=file_)
                g_idx += 1
                plt.ylabel("N")
            ax.set_xlim(0, 700)
            ax.set_ylim(-3.0, 3.0)
        plt.legend(loc='upper center', bbox_to_anchor=(-0.15, -0.15),
                  ncol=3, fancybox=True, shadow=True)

        plt.savefig(file_name_force)
        l_idx = 0
        g_idx = 0

        names = ['Vx', 'Wx', 'Vy', 'Wy', 'Vz', 'Wz']
        subplot_idx = 320 
        plt.figure("l_vel - a_vel", figsize=(15, 15))
        # plt.clf()
        for j in range(6):
            subplot_idx += 1
            ax = plt.subplot(subplot_idx)
            ax.set_title(names[j])
            plt.xlabel("num data")
            if j%2 == 1:
                ax.plot(lpf(a_vel[:,l_idx]), label=file_)
                l_idx += 1
                plt.ylabel("rad/s")
                # if j == 1:
                #     plt.legend(replay_data_files)
            else:
                plt.plot(lpf(l_vel[:,g_idx]), label=file_)
                g_idx += 1
                plt.ylabel("m/s")
            ax.set_xlim(0, 700)
            ax.set_ylim(-1.0, 1.0)

        plt.legend(loc='upper center', bbox_to_anchor=(-0.15, -0.15),
                  ncol=3, fancybox=True, shadow=True)

        plt.savefig(file_name_traj)

        # plt.close(title)
        plt.draw()
        plt.pause(0.00001)
        # raw_input()
        # if k == 'q':
            # break


if __name__ == '__main__':
    visualise_data()