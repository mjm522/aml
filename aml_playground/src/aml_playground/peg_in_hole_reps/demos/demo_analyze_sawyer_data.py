import os
import copy
import pandas
import numpy as np
import quaternion as q
import scipy.signal as sig
import matplotlib.pyplot as plt
from aml_io.io_tools import load_data
from scipy.interpolate import interp1d
# from sklearn.mixture import GaussianMixture
from file_paths_spring import replay_data_path, replay_data_files
from aml_playground.peg_in_hole_reps.utilities.draw_frame import draw_frame
from aml_playground.peg_in_hole_reps.utilities.utils import convert_list_str_ft_reading
# from aml_playground.peg_in_hole_reps.utilities.visualization import visualize_3d_gmm, visualize_2d_gmm

# replay_data_path = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/sawyer/spring_exp_1/'
# replay_data_files = ['spring_exp_1.pkl']

# replay_data_path = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/sawyer/replayed_data/data_with_ft_test/'
# /home/br/catkin_ws/baxter_ws/src/aml/aml_data/aml_lfd/sawyer_replayed_data_ft_set/side2_fw/90deg_04.pkl
# replay_data_files = ['90deg_03.pkl'#,
#                     # 'sawyer_replayed_data_side_1_bkwd.pkl',
#                     #  'sawyer_replayed_data_side_2_bkwd.pkl',
#                     #  'sawyer_replayed_data_side_3_bkwd.pkl',
#                     #  'sawyer_replayed_data_side_4_bkwd.pkl',
#                     #  'sawyer_replayed_data_side_1_fwd.pkl',
#                     #  'sawyer_replayed_data_side_2_fwd.pkl',
#                     #  'sawyer_replayed_data_side_3_fwd.pkl',
#                     #  'sawyer_replayed_data_side_4_fwd.pkl'
#                      ]



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
    rot_frame_list = []

    for j in range(len(force_list)):

        trans_force_l= []
        force_l = force_list[j]
        pos_l = pos_list[j]
        ori_l = ori_list[j]
        rot_frames_l = []

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

            rot_frames_l.append(copy.deepcopy(rot_contact_frame_T.T))
     
        trans_force_list.append(np.asarray(trans_force_l))
        rot_frame_list.append(np.asarray(rot_frames_l))

    return trans_force_list, rot_frame_list

def plot_(right_data,
          left_data,
          f_title="None",
          p_title='None',
          names=['dummyx', 'dummyx', 'dummyy', 'dummyy', 'dummyz', 'dummyz'],
          do_clf=False,
          ylabels=['none', 'none'],
          r_lpf=True,
          l_lpf=True):
    
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
            if l_lpf:
                ax.plot(lpf(left_data[:,l_idx]), label=p_title)
            else:
                ax.plot(left_data[:,l_idx], label=p_title)
            l_idx += 1
            plt.ylabel(ylabels[0])
        elif j%2 == 1:
            if r_lpf:
                plt.plot(lpf(right_data[:,r_idx]), label=p_title)
            else:
                ax.plot(right_data[:,r_idx], label=p_title)
            r_idx += 1
            plt.ylabel(ylabels[1])

        # ax.set_xlim(0, max(left_data.shape[0], right_data.shape[0]))
        # ax.set_ylim(-3.0, 3.0)
    plt.legend(loc='upper center', bbox_to_anchor=(-0.15, -0.15),
              ncol=3, fancybox=True, shadow=True)

    return fig

def plot_3D(force_list, pos_list, ori_list, interval=30, title=None):
    
    fig = plt.figure(title, figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')

    for k in range(len(ori_list)):

        if k%interval==0:
            if pos_list is None:
                pos = np.array([0.,0.,0.])
            else:
                pos = pos_list[k]

            draw_frame(pos=pos, ori=ori_list[k], axes=ax, l=0.15)

    return fig

def time2collidedata(force_list, pos_list, ori_list, vel_list, ang_vel_list):

    num_lists = len(force_list)
    plt.ion()

    def time2collide():

        time_2_collide = np.zeros(num_lists)

        for l in range(num_lists):

            force_l = np.vstack([lpf(force_list[l][:,0]),
                                 lpf(force_list[l][:,1]),
                                 lpf(force_list[l][:,2])]).T

            for k in range(len(force_l)):
                if np.linalg.norm(force_l[k, :]) > 0.17:
                    time_2_collide[l] = int(k)
                    break

        return time_2_collide.astype("int")

    time_2_collide = time2collide()

    for j in range(num_lists):

        force_l = force_list[j][:time_2_collide[j]+5, :]
        
        vel_l = np.asarray(vel_list[j])[:time_2_collide[j]+5, :]

        plot_(right_data=force_l,
          left_data=vel_l,
          f_title="Force-Velocity",
          p_title='Force-Velocity till Collision+5',
          names=['Fx', 'Vx', 'Fy', 'Vy', 'Fz', 'Vz'],
          do_clf=False,
          ylabels=['N', 'm/s'],
          l_lpf=True,
          r_lpf=False)

        plt.draw()
        plt.pause(0.00001)
        raw_input()

def fit_gmm(force_list):

    force_list = np.asarray(force_list)
    points_x, points_y, points_z = np.array([]),np.array([]),np.array([])
    # new_x = np.array([])

    for l in range(len(force_list)):

        points_x = np.hstack([points_x, lpf(force_list[l][:,0])])
        points_y = np.hstack([points_y, lpf(force_list[l][:,1])])
        points_z = np.hstack([points_z, lpf(force_list[l][:,2])])

        if l == 0:
            data_to_fit = np.array(lpf(force_list[l][:,x]).T)
        else:
            data_to_fit = np.vstack([data_to_fit, lpf(force_list[l][:,x]).T])
        # print lpf(force_list[l][:,0]).T.shape

    points = np.vstack([points_x, points_y, points_z]).T
    # print data_to_fit.shape
    # raw_input()

    # lowest_bic = np.infty
    # bic = []
    # n_components_range = range(1, 100)
    # cv_types = ['spherical', 'tied', 'diag', 'full']
    
    # for cv_type in cv_types:
    #     print "Testing cvtype \t:", cv_type
    #     # for n_components in n_components_range:
    #     # print "Test with %d components"%(n_components)
    #     # Fit a Gaussian mixture with EM
    #     gmm = GaussianMixture(n_components=99,
    #                           covariance_type='diag')
    #     gmm.fit(points)
    #     bic.append(gmm.bic(points))
    #     if bic[-1] < lowest_bic:
    #         lowest_bic = bic[-1]
    #         best_gmm = gmm

    best_gmm = GaussianMixture(n_components=int(len(force_list)/10),
                              covariance_type='diag')
    best_gmm.fit(data_to_fit)

    for l in range(data_to_fit.shape[0]):
        if l%10 == 0:
            print " "
        print int(l/10), best_gmm.predict(data_to_fit[l,:].reshape(1, -1))

    print best_gmm.means_.shape

    # visualize_3d_gmm(points, best_gmm.weights_, best_gmm.means_.T, np.sqrt(best_gmm.covariances_).T)

def resample_data(force_list, pos_list, ori_list, num_resamples=500, do_lpf=False):

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
            if do_lpf:
                resampled_data[:,k] = interp1d(osample, lpf(total_data[:,k]), kind='cubic', fill_value='extrapolate')(nsample)
            else:
                resampled_data[:,k] = interp1d(osample, total_data[:,k], kind='cubic', fill_value='extrapolate')(nsample)

        #re-normalize quaternions
        for i in range(num_resamples):
            resampled_data[i,:-4] /= np.linalg.norm(resampled_data[i,:-4])

        resampled_force_list.append(copy.deepcopy(resampled_data[:, :3]))
        resampled_pos_list.append(copy.deepcopy(resampled_data[:, 3:6]))
        resampled_ori_list.append(copy.deepcopy(resampled_data[:, 6:]))

    return resampled_force_list, resampled_pos_list, resampled_ori_list

def visualise_data(file_='right2left', show_plots = True): #right2left left2right

    file_list = replay_data_files
    data_path = replay_data_path#replay_data_path

    #contacts_data_left2right_changing_angles
    # file = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/contacts_data_'+file_+'_changing_angles.pkl'

    pos_list, ori_list, lin_vels_list, ang_vels_list, f_list, t_list, pos_list = [],[],[],[],[],[],[]

    for data_file in file_list:

        file = data_path + data_file
        data = load_data(file)[0].get_contents()
        N = len(data)

        pos_list.append([data[i]['tip_state']['position'] for i in range(N)])
        ori_list.append([data[i]['tip_state']['orientation'] for i in range(N)])
        lin_vels_list.append([data[i]['tip_state']['linear_vel'] for i in range(N)])
        ang_vels_list.append([data[i]['tip_state']['angular_vel'] for i in range(N)])

        ft_reading = convert_list_str_ft_reading([data[i]['ft_reading'] for i in range(N)])
        f_list.append(ft_reading[:, :3])
        t_list.append(ft_reading[:, 3:])

    # time2collidedata(f_list, pos_list, ori_list, lin_vels_list, ang_vels_list)
    trans_force_list_before_resample, rot_frame_list = transform_forces(f_list, pos_list, ori_list)
    resampled_force_list, resampled_pos_list,  resampled_ori_list = resample_data(f_list, pos_list, ori_list, num_resamples=500, do_lpf=True)
    trans_force_list_after_resample, _ = transform_forces(resampled_force_list, resampled_pos_list,  resampled_ori_list)
    # fig_ee = plot_3D(f_list[0], pos_list[0], ori_list[0], interval=30, title="ee")
    # fig_mv = plot_3D(None, pos_list[0], rot_frame_list[0], interval=30, title="mv")
    # plt.show()
    # fit_gmm(trans_force_list_after_resample)

    # # print data[0].get_contents()
    # # print "done"
    # raw_input()

    # data_to_show = [0,5,10,15,20]

    # col = ['r','g','b','y','c']
    # plt.figure("force")
    # plt.figure("traj")

    if show_plots:
        plt.ion()

        for k in range(len(file_list)):

            print k+1, file_list[k]

            file_folder = str(file_list[k].split('/')[0])
            file_name   = str(('_'.join(file_list[k].split('_')[3:])).split('.')[0])

            file_folder = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/saif_experiments/' + file_folder

            if not os.path.exists(file_folder):
                os.makedirs(file_folder)

            file_name_force = file_folder +'/'+ file_name + '_images_force_' + str(k)+ '.png'

            # file_name_traj = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/' + file_ +'_images_traj_' + str(k)+ '.png'

            # fig_force = plot_(left_data=np.asarray(f_list[k]),
            #   right_data=trans_force_list_before_resample[k],
            #   f_title="force - trans-force",
            #   p_title=file_name,
            #   names=['Fx', 'tranFx', 'Fy', 'tranFy', 'Fz', 'tranFz'],
            #   do_clf=False,
            #   ylabels=['N', 'N'])

            fig_force = plot_(left_data=np.asarray(pos_list[k]),
              right_data=trans_force_list_before_resample[k],
              f_title="force - extension",
              p_title=file_name,
              names=['Fx', 'Px', 'Fy', 'Py', 'Fz', 'Pz'],
              do_clf=False,
              ylabels=['N', 'N'])

            # fig_force = plot_(right_data=trans_force_list_after_resample[k],
            #   left_data=np.asarray(trans_force_list_before_resample[k]),
            #   f_title="force - trans-force",
            #   p_title=file_,
            #   names=['tranFx', 'R-tranFx', 'tranFy', 'R-tranFy', 'tranFz', 'R-tranFz'],
            #   do_clf=False,
            #   ylabels=['N', 'N'],
            #   l_lpf=True,
            #   r_lpf=True)

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

            # plt.savefig(file_name_force)
            plt.draw()
            plt.pause(0.00001)
            # raw_input()

            if (k > 0) and ((k+1)%10==0):
                raw_input()
                plt.close("force - extension")

def save_to_csv():

    file_list = replay_data_files
    data_path = replay_data_path#replay_data_path

    #contacts_data_left2right_changing_angles
    # file = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/contacts_data_'+file_+'_changing_angles.pkl'

    pos_list, ori_list, lin_vels_list, ang_vels_list, f_list, t_list, pos_list = [],[],[],[],[],[],[]

    # full_ft = np.asarray([])
    file_count = 0
    for data_file in file_list:

        print data_file

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

        # print ft_reading
        if file_count == 0:
            full_ft = copy.deepcopy(ft_reading)
        else:
            full_ft = np.vstack([full_ft,np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])])
            full_ft = np.vstack([full_ft,ft_reading])

        # print full_ft
        # print full_ft.shape

        file_count+=1

        # raw_input()
    print full_ft.shape

    savepath = data_path + 'side2_fw.csv'
    np.savetxt(savepath, full_ft, delimiter=',')

        # print f_list[0]
        # raw_input()


if __name__ == '__main__':
    # save_to_csv()
    visualise_data()