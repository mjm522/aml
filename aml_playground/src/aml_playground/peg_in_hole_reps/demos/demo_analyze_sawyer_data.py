import os
import copy
import pandas
import numpy as np
import quaternion as q
import scipy.signal as sig
import matplotlib.pyplot as plt
from aml_io.io_tools import load_data
from scipy.interpolate import interp1d
from aml_playground.peg_in_hole_reps.utilities.draw_frame import draw_frame

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


def transform_forces(force_list, pos_list, ori_list):

    trans_force_list = []

    for j in range(len(force_list)):

        trans_force_l= []
        force_l = force_list[j]
        pos_l = pos_list[j]
        ori_l = ori_list[j]

        for k in range(len(pos_l)-1):
            point_1 = pos_l[k]
            point_2 = pos_l[k+1]
            ori = ori_l[k]

            rot_sensor_frame = q.as_rotation_matrix(q.quaternion(ori[0], ori[1], ori[2], ori[3]))

            x_axis  = point_2-point_1
            x_axis[x_axis < 1e-5] = 0.
            if np.linalg.norm(x_axis) < 1e-5:
                x_axis = np.array([0.,0.,0])
            else:
                x_axis  /= np.linalg.norm(x_axis)
            z_axis  = np.array([0.,0.,1.])
            y_axis  = np.cross(z_axis, x_axis)

            rot_contact_frame_T  = np.vstack([x_axis, y_axis, z_axis])

            trans_force_l.append(np.dot(rot_contact_frame_T, np.dot(rot_sensor_frame, force_l[k,:])))
     
        trans_force_list.append(np.asarray(trans_force_l))

    return trans_force_list


def plot_(right_data,
          left_data,
          f_title="None",
          p_title='None',
          names=['dummyx', 'dummyx', 'dummyy', 'dummyy', 'dummyz', 'dummyz'],
          do_clf=False,
          ylabels=['none', 'none']):
    
    num_plots = len(names)
    
    if num_plots == 8:
        subplot_idx = 420
    elif num_plots == 6:
        subplot_idx = 320

    l_idx, r_idx = 0, 0  
    fig = plt.figure(f_title, figsize=(15, 15))
    plt.title(p_title)
    
    if do_clf:
        plt.clf()
    
    for j in range(num_plots):
        
        subplot_idx += 1
        ax = plt.subplot(subplot_idx)
        ax.set_title(names[j])
        plt.xlabel("num data")
        if j%2 == 0:
            ax.plot(lpf(left_data[:,l_idx]), label=p_title)
            l_idx += 1
            plt.ylabel(ylabels[0])
        elif j%2 == 1:
            plt.plot(lpf(right_data[:,r_idx]), label=p_title)
            r_idx += 1
            plt.ylabel(ylabels[1])
        ax.set_xlim(0, max(left_data.shape[0], right_data.shape[0]))
        ax.set_ylim(-3.0, 3.0)
    plt.legend(loc='upper center', bbox_to_anchor=(-0.15, -0.15),
              ncol=3, fancybox=True, shadow=True)

    return fig



def plot_3D(force_list, pos_list, ori_list):
    
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')

    for k in range(len(pos_list)):

        if k%30==0:
            draw_frame(pos=pos_list[k], ori=ori_list[k], axes=ax, l=0.15)

    plt.show()


def resample_data(force_list, pos_list, ori_list, num_resamples=500):

    resampled_force_list, resampled_pos_list, resampled_ori_list = [],[],[]

    for j in range(len(pos_list)):

        total_data = np.hstack([force_list[j], pos_list[j], ori_list[j]])
        num_samples, num_dim = total_data.shape
        resampled_data = np.zeros([num_resamples, num_dim])
        #some of them has less number of data
        #but how to upsample then?
    
        for k in range(num_dim):
            osample  = np.arange(0.,  1., 1./num_samples)[:num_samples]
            nsample  = np.arange(0.,  1., 1./num_resamples)
            resampled_data[:,k] = interp1d(osample, total_data[:,k], kind='cubic', fill_value='extrapolate')(nsample)

        #re-normalize quaternions
        for i in range(num_resamples):
            resampled_data[i,:-4] /= np.linalg.norm(resampled_data[i,:-4])

        resampled_force_list.append(copy.deepcopy(resampled_data[:, :3]))
        resampled_pos_list.append(copy.deepcopy(resampled_data[:, 3:6]))
        resampled_ori_list.append(copy.deepcopy(resampled_data[:, 6:]))

    return resampled_force_list, resampled_pos_list, resampled_ori_list


def visualise_data(file_='right2left'): #right2left left2right

    file_list = replay_data_files
    data_path = replay_data_path#replay_data_path

    #contacts_data_left2right_changing_angles
    # file = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/contacts_data_'+file_+'_changing_angles.pkl'

    pos_list, ori_list, lin_vels_list, ang_vels_list, f_list, t_list, pos_list = [],[],[],[],[],[],[]

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
        #     f_list.append(f_re           ading)
        #     t_list.append(t_reading)
        # pos_list.append([data[i]['tip_state']['position'] for i in range(len(data))])

    trans_force_list_before_resample = transform_forces(f_list, pos_list, ori_list)
    resampled_force_list, resampled_pos_list,  resampled_ori_list = resample_data(f_list, pos_list, ori_list, num_resamples=1000)
    trans_force_list_after_resample = transform_forces(resampled_force_list, resampled_pos_list,  resampled_ori_list)

    # print data[0].get_contents()
    # print "done"
    # raw_input()

    # data_to_show = [0,5,10,15,20]

    # col = ['r','g','b','y','c']
    # plt.figure("force")
    # plt.figure("traj")
    plt.ion()

    for k in range(len(file_list)):

        print k+1, file_list[k]

        file_ = str(('_'.join(file_list[k].split('_')[3:])).split('.')[0])

        file_name_force = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/' + file_+ '_images_force_' + str(k)+ '.png'
        file_name_traj = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/' + file_ +'_images_traj_' + str(k)+ '.png'

        # fig_force = plot_(right_data=trans_force_list_before_resample[k],
        #   left_data=np.asarray(f_list[k]),
        #   f_title="force - trans-force",
        #   p_title=file_,
        #   names=['Fx', 'tranFx', 'Fy', 'tranFy', 'Fz', 'tranFz'],
        #   do_clf=False,
        #   ylabels=['N', 'N'])

        fig_force = plot_(right_data=trans_force_list_after_resample[k],
          left_data=np.asarray(trans_force_list_before_resample[k]),
          f_title="force - trans-force",
          p_title=file_,
          names=['tranFx', 'R-tranFx', 'tranFy', 'R-tranFy', 'tranFz', 'R-tranFz'],
          do_clf=False,
          ylabels=['N', 'N'])

        # fig_vel = plot_(right_data=np.asarray(ang_vels_list[k]),
        #   left_data=np.asarray(lin_vels_list[k]),
        #   f_title="lin-vel <=> ang-vel",
        #   p_title=file_,
        #   names=['Vx', 'Wx', 'Vy', 'Wy', 'Vz', 'Wz'],
        #   do_clf=False,
        #   ylabels=['m/s', 'rad/s'])

        # fig_force_re = plot_(right_data=resampled_force_list[k],
        #                   left_data=np.asarray(f_list[k]),
        #                   f_title="sample - resamples - force",
        #                   p_title=file_,
        #                   names=['fx', 'R-fx', 'fy', 'R-fy', 'fz', 'R-fz'],
        #                   do_clf=False,
        #                   ylabels=['N', 'N'])

        # fig_pos_re = plot_(right_data=resampled_pos_list[k],
        #                   left_data=np.asarray(pos_list[k]),
        #                   f_title="sample - resamples - pos",
        #                   p_title=file_,
        #                   names=['x', 'R-x', 'y', 'R-y', 'z', 'R-z'],
        #                   do_clf=False,
        #                   ylabels=['m', 'm'])

        # fig_ori_re = plot_(right_data=resampled_ori_list[k],
        #           left_data=np.asarray(ori_list[k]),
        #           f_title="sample - resamples - ori",
        #           p_title=file_,
        #           names=['x', 'R-x', 'y', 'R-y', 'z', 'R-z', 'w', 'R-w'],
        #           do_clf=False,
        #           ylabels=['none', 'none'])

        # plt.savefig(file_name_traj)
        plt.draw()
        plt.pause(0.00001)
        raw_input()


if __name__ == '__main__':
    visualise_data()