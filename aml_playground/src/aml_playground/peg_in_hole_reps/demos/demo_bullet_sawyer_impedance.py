import os
import time
import rospy
import copy
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt
from aml_io.io_tools import save_data, load_data
from aml_ctrl.utilities.min_jerk_interp import MinJerkInterp
from aml_lfd.utilities.smooth_demo_traj import SmoothDemoTraj
from aml_rl_envs.utils.collect_demo import plot_demo, draw_trajectory
from aml_playground.peg_in_hole_reps.controller.sawyer_imp_reps import SawyerImpREPS
#get the experiment params
from aml_rl_envs.utils.collect_demo import plot_demo
from aml_rl_envs.utils.data_utils import save_csv_data
from aml_playground.peg_in_hole_reps.exp_params.experiment_imp_params import exp_params


def ray_test(env):

    ee_pos, ee_ori = env._sawyer.get_ee_pose(as_tuple=False)

    final_pt = copy.deepcopy(ee_pos)
    final_pt[2] = final_pt[2] - 1.

    result = pb.rayTest(rayFromPosition=ee_pos,
                        rayToPosition=final_pt,
                        physicsClientId=env._cid
                        )
    if result[0][0] == env._table_id:
        print "Found a point on the table!" 
        print "Projection on the table \t", result[0][3]

    return np.asarray(result[0][3])

def make_demo(y_dir=1):
    #y_dir = 1 to draw from left to right when facing sawyer (sawyer._jnt_postns[0]=0.15703139)
    #y_dir = -1 to draw from right to left when facing sawyer (sawyer._jnt_postns[0]=0.15703139-np.pi/2)

    ps = SawyerImpREPS(False, exp_params)
    env = ps._eval_env
    file_name = os.environ['AML_DATA'] + '/aml_lfd/right_sawyer_exp_peg_in_hole/sawyer_bullet_ee_states_imp3.csv'
    start_location, ee_ori = env._sawyer.get_ee_pose()
    table_location = ray_test(env) #np.array([0.4631, -0.3410, -0.0623])#np.array([0.6903, -0.0862, -0.064]) #-0.3410,
    middle_location = table_location + np.array([0.15, y_dir*0.15, 0.])
    end_location = table_location + np.array([0.45, y_dir*0.45, 0.])
    # start_location = table_location - np.array([0.25, -0.45, -0.25])
    # middle_location = table_location - np.array([0.15, -0.35, 0])
    # end_location = table_location + np.array([0.15, -0.15, 0.])
    minjerk   = MinJerkInterp()
    
    minjerk.configure(start_pos=start_location, goal_pos=middle_location)
    min_jerk_traj_1 = minjerk.get_interpolated_trajectory()

    minjerk.configure(start_pos=middle_location, goal_pos=end_location)
    min_jerk_traj_2 = minjerk.get_interpolated_trajectory()

    ee_demo = np.vstack([min_jerk_traj_1['pos_traj'], min_jerk_traj_2['pos_traj']])

    # smoother = SmoothDemoTraj(ee_demo, 109)
    # ee_demo = smoother._smoothed_traj

    plot_demo(ee_demo, color=[1,0,0], start_idx=0, life_time=0., cid=env._cid)
    save_csv_data(file_name, ee_demo)

    raw_input("press any key to exit")

def plot_data(z_dir=1):

    #left to right z_dir = 1
    #right to left z_dir = 0

    ps = SawyerImpREPS(False, exp_params)

    contacts_data_list = []

    # print 'ori', ps._eval_env._sawyer.state()['ee_ori']
    # (-0.0096177, 0.02913485, 0.70694985, -0.70659788)
    # ori = pb.getEulerFromQuaternion((-0.0096177, 0.02913485, 0.70694985, -0.70659788))
    
    pos = ps._eval_env._sawyer.state()['ee_point']
    cmd = ps._eval_env._sawyer._jnt_postns

    # 1. [ 0.15703249 -1.64921352  0.38449968  1.98040024  0.01590287  1.21045616 2.33552743]
    
    # ee_ori_list = [ (-1.57205067-np.pi/2, 0.02757827, 3.08675889), (-1.15350192-np.pi/2, 0.24331851, -2.97215963), (-1.92222604-np.pi/2, -0.2374129, 2.96860768)]
    # j_pos_list = [[0.15703139, -1.64921782,  0.3845158,   1.98039767,  0.01590169,  1.21045897, 2.33553496], 
    #               [0.15703139, -1.64921782, 0.3845158, 1.98039767, -0.47909830999996356, 1.21045897, 2.33553496],
    #               [0.15703139, -1.64921782, 0.3845158, 1.98039767, 0.47909830999996356, 1.21045897, 2.33553496]]
    j_pos_list = [[0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, -0.43119830999996356, 1.21045897, 2.33553496], 
                  [0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, -0.38329830999996356, 1.21045897, 2.33553496], 
                  [0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, -0.33539830999996356, 1.21045897, 2.33553496], 
                  [0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, -0.28749830999996356, 1.21045897, 2.33553496], 
                  [0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, -0.23959830999996357, 1.21045897, 2.33553496], 
                  [0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, -0.19169830999996357, 1.21045897, 2.33553496], 
                  [0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, -0.14379830999996357, 1.21045897, 2.33553496], 
                  [0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, -0.09589830999996357, 1.21045897, 2.33553496], 
                  [0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, -0.04799830999996357, 1.21045897, 2.33553496], 
                  [0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, -9.830999996357503e-05, 1.21045897, 2.33553496], 
                  [0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, 0.04780169000003642, 1.21045897, 2.33553496], 
                  [0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, 0.09570169000003642, 1.21045897, 2.33553496], 
                  [0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, 0.14360169000003642, 1.21045897, 2.33553496], 
                  [0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, 0.19150169000003642, 1.21045897, 2.33553496], 
                  [0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, 0.23940169000003642, 1.21045897, 2.33553496], 
                  [0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, 0.2873016900000364, 1.21045897, 2.33553496], 
                  [0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, 0.3352016900000364, 1.21045897, 2.33553496], 
                  [0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, 0.3831016900000364, 1.21045897, 2.33553496], 
                  [0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, 0.4310016900000364, 1.21045897, 2.33553496], 
                  [0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, 0.4789016900000364, 1.21045897, 2.33553496], 
                  [0.15703139-z_dir*np.pi/2, -1.64921782, 0.3845158, 1.98039767, 0.5268016900000364, 1.21045897, 2.33553496]]

    ee_ori_list = [(-1.1967808361505157-np.pi/2, 0.22681123949843532, -2.998070623273961+z_dir*1.5*np.pi), 
                   (-1.2394452229333248-np.pi/2, 0.209236720092549, -3.023129896726811+z_dir*1.5*np.pi), 
                   (-1.2814919423138877-np.pi/2, 0.19064200410284454, -3.0473115758150775+z_dir*1.5*np.pi), 
                   (-1.3229239258703487-np.pi/2, 0.17107456178452923, -3.0705947391061224+z_dir*1.5*np.pi), 
                   (-1.3637497677939128-np.pi/2, 0.15058195825886586, -3.09296288879215+z_dir*1.5*np.pi), 
                   (-1.4039832997714248-np.pi/2, 0.12921157667879327, -3.1144033922021936+z_dir*1.5*np.pi), 
                   (-1.443643161911009-np.pi/2, 0.10701038626075328, -3.1349069074360503+z_dir*1.5*np.pi), 
                   (-1.4827523844979498-np.pi/2, 0.08402475407086263, 3.128718500742748+z_dir*1.5*np.pi), 
                   (-1.5213379926455433-np.pi/2, 0.060300298583515974, 3.1101067012483212+z_dir*1.5*np.pi), 
                   (-1.5594306433135674-np.pi/2, 0.035881782437549026, 3.0924458932894887+z_dir*1.5*np.pi), 
                   (-1.597064301820933-np.pi/2, 0.010813041467977827, 3.0757379105696017+z_dir*1.5*np.pi), 
                   (-1.6342759629484642-np.pi/2, -0.014863053046756338, 3.0599840379193712+z_dir*1.5*np.pi), 
                   (-1.6711054200439985-np.pi/2, -0.04110460194924399, 3.0451854963744696+z_dir*1.5*np.pi), 
                   (-1.70759508420713-np.pi/2, -0.0678706584388984, 3.031343904572704+z_dir*1.5*np.pi), 
                   (-1.7437898546258275-np.pi/2, -0.0951211835688976, 3.0184617184622797+z_dir*1.5*np.pi), 
                   (-1.7797370404290214-np.pi/2, -0.12281697924962938, 3.006542652328842+z_dir*1.5*np.pi), 
                   (-1.8154863339668086-np.pi/2, -0.1509196006071234, 2.9955920849198+z_dir*1.5*np.pi), 
                   (-1.8510898351877276-np.pi/2, -0.17939124911276247, 2.9856174550051597+z_dir*1.5*np.pi), 
                   (-1.8866021267031499-np.pi/2, -0.2081946474454253, 2.976628651095955+z_dir*1.5*np.pi), 
                   (-1.9220803991637967-np.pi/2, -0.2372928965772673, 2.9686384002687047+z_dir*1.5*np.pi), 
                   (-1.9575846266727521-np.pi/2, -0.2666493150945791, 2.9616626611315504+z_dir*1.5*np.pi)]


    old_cmd = np.zeros(7)

    done = False

    # init_j_pos = [0.15703139, -1.64921782, 0.3845158, 1.98039767, -0.47909830999996356, 1.21045897, 2.33553496]

    # ee_oris = []
    # j_poses = []
    # i = 1
    # import copy
    # while init_j_pos[4] < 0.48:

    # #     # cmd = ps._eval_env._sawyer.inv_kin(ee_pos=pos.tolist(), ee_ori=ee_ori_list[2])
    # #     cmd[4] += 0.0001
    # #     ps._eval_env._sawyer.apply_action(cmd)
    #     init_j_pos[4] += 0.0479
    #     ps._eval_env._reset(jnt_pos = init_j_pos)
    #     # ps._eval_env._sawyer.simple_step()
    # #     diff = np.linalg.norm(ee_ori_list[1] - ps._eval_env._sawyer.state(ori_type = 'eul')['ee_ori'])
    # #     print diff, '   cmd: ', cmd

    # #     done = diff < 0.0001

    #     # old_cmd = cmd.copy()

    # # print "Cmd \t", cmd
    #     print i
    #     i+=1
    #     print init_j_pos

    #     pos_ = copy.deepcopy(init_j_pos)
    #     ori = ps._eval_env._sawyer.state(ori_type = 'eul')['ee_ori']
    #     print ori
    #     ee_oris.append((ori[0],ori[1],ori[2]))
    #     j_poses.append(pos_)
    # # raw_input()
    # print "  "
    # print j_poses
    # print ee_oris
    # raw_input()
    num = 15
    lf_list = np.linspace(0.,2., num)
    # col = ['r','g','b']
    for j in range(1):

        data_list = []
        
        lf = 0.45

        title = "Friction-"+str(round(lf,3))
        # plt.figure(title)
        # plt.ion()

        for k, ee_ori in enumerate(ee_ori_list):

            ps._eval_env._reset(lf=lf, jnt_pos = j_pos_list[k])

            # ee_pos, ee_ori_tmp = ps._eval_env._sawyer.get_ee_pose(as_tuple=False)

            # print "\n\n\n\n\n\n\n"
            # print k
            # print pb.getEulerFromQuaternion(ee_ori_tmp)
            # print "\n\n\n\n\n\n\n"

            # raw_input()

            # continue

            data = ps.draw(ee_ori=ee_ori)

            data['lateral_fc'] = lf

            data_list.append(data)
            
            # ps._eval_env.reward(data)
            
            # ee_traj  = data['ee_traj']
            # ee_wrenches = data['ee_wrenches']
            # ee_wrenches_local = data['ee_wrenches_local']

            # # print data['contact_details']
            # # contact_forces_normal = []
            # # for i in data['contact_details']:
            # #     contact_forces_normal.append(i[0]['contact_force'] if len(i)>0 else 0)

            # forces_list  = ee_wrenches[:,:3]
            # torques_list = ee_wrenches[:,3:]

            # forces_list_local  = ee_wrenches_local[:,:3]
            # torques_list_local = ee_wrenches_local[:,3:]

            # l_idx = 0
            # g_idx = 0
                    
            # subplot_idx = 320   
            # color = col[k]
            # for j in range(6):
            #     subplot_idx += 1
            #     # plt.figure("force")
            #     plt.subplot(subplot_idx)
            #     if j%2 == 1:
            #         plt.plot(forces_list_local[:,l_idx], color)
            #         l_idx += 1
            #     else:
            #         plt.plot(forces_list[:,g_idx], color)
            #         g_idx += 1
                # plt.figure("traj")
                # plt.subplot(subplot_idx)
                # plt.plot(ee_traj[:,k], color)
            # raw_input()
            # plot_demo(data['dmp'], start_idx=0, life_time=0, color=(1,0,0))
            # plot_demo(ee_traj, start_idx=0, life_time=0, color=[0,1,0])
            # raw_input()
            # plt.plot(contact_forces_normal, col[k])

        contacts_data_list.append(data_list)

        # plt.draw()
        # plt.pause(0.00001)

        # raw_input()
        # file_name = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/' + title +str(k)+ '_clean.png'
        # plt.savefig(file_name)
        # plt.close(title)
        data_name = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/contacts_data_imp2_' + str(round(lf,3)) + '.pkl'
        save_data(filename=data_name, data=contacts_data_list)

def check_s_w(joint_space=False):

    ps = SawyerImpREPS(joint_space, exp_params)

    s = 0.5 - 6.645408120672127716e-02

    w_def = None
    w = np.array([-0.5, 0.5, -0.5, -0.33817151, -0.5, 0.5])
    
    ws = [w_def, w]

    titles = ['without_params', 'with_params']

    for w, title in zip(ws, titles):

        plt.figure(title)
        ps._eval_env._reset(lf=s)

        data, reward = ps._eval_env.execute_policy(w=w,s=s)
        
        ee_wrenches = data['ee_wrenches']

        forces_list  = ee_wrenches[:,:3]

        torques_list = ee_wrenches[:,3:]
                
        subplot_idx = 310

        for k in range(3):
            subplot_idx += 1
            plt.subplot(subplot_idx)
            plt.plot(forces_list[:,k])

    plt.show()

def reps(joint_space=False):

    rewards = []
    params  = []
    force_penalties = []
    goal_penalties = []

    ps = SawyerImpREPS(joint_space, exp_params)

    # make_demo(ps._sim_env)

    plt.figure("Reward plots", figsize=(15,15))
    plt.ion()

    for i in range(100):

        print "Episode \t", i

        s = ps._eval_env.context()

        policy = ps._gpreps.run()

        w = policy.compute_w(s, transform=True, explore=False)

        _, reward = ps._eval_env.execute_policy(w, s, show_demo=False)

        mean_reward = ps._eval_env._penalty['total']
        force_penalty = ps._eval_env._penalty['force']

        print "Parameter found*****************************************: \t", w
        print "mean_reward \t", mean_reward

        if mean_reward > -1000:
            rewards.append(mean_reward)

        force_penalties.append(force_penalty)

        goal_penalties.append(force_penalty+mean_reward)
        
        params.append(np.hstack([w,s,mean_reward]))

        ps._eval_env._reset()
        plt.clf()
        plt.subplot(311)
        plt.title('Total reward')
        plt.plot(rewards, 'r')
        plt.ylabel("mag")
        plt.subplot(312)
        plt.title('Force felt')
        plt.plot(force_penalties, 'g')
        plt.ylabel("mag")
        plt.subplot(313)
        plt.title('Goal Closeness')
        plt.plot(goal_penalties, 'b')
        plt.xlabel("iterations")
        plt.ylabel("mag")
        plt.pause(0.00001)
        plt.draw()

    file_name = os.environ['AML_DATA'] + '/aml_lfd/right_sawyer_exp_peg_in_hole/params.csv'
    save_csv_data(file_name, np.asarray(params))
    raw_input("Press any key to exit")

def visualise_data(file_='right2left'): #right2left left2right

    #contacts_data_left2right_changing_angles

    file = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/contacts_data_'+file_+'_changing_angles.pkl'

    data = load_data(file)[0]

    # data_to_show = [0,5,10,15,20]

    # col = ['r','g','b','y','c']
    plt.figure("force")
    plt.figure("traj")
    plt.ion()

    for k, i in enumerate(data):

        ee_traj  = i['ee_traj']
        ee_wrenches = i['ee_wrenches']
        ee_wrenches_local = i['ee_wrenches_local']  

        forces_list  = ee_wrenches[:,:3]
        torques_list = ee_wrenches[:,3:]

        forces_list_local  = ee_wrenches_local[:,:3]
        torques_list_local = ee_wrenches_local[:,3:]

        l_idx = 0
        g_idx = 0
        
        file_name_force = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/' + file_+ '_images/force_' + str(k)+ '.png'
        file_name_traj = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/' + file_ +'_images/traj_' + str(k)+ '.png'

        subplot_idx = 320   
        # color = col[k]
        plt.figure("force")
        for j in range(6):
            subplot_idx += 1
            plt.subplot(subplot_idx)
            if j%2 == 1:
                plt.plot(forces_list_local[:,l_idx])
                l_idx += 1
            else:
                plt.plot(forces_list[:,g_idx])
                g_idx += 1

        plt.savefig(file_name_force)

        subplot_idx = 310
        plt.figure("traj")
        for j in range(3):
            subplot_idx += 1
            plt.subplot(subplot_idx)
            plt.plot(ee_traj[:,j])

        plt.savefig(file_name_traj)

        # plt.close(title)
        plt.draw()
        plt.pause(0.00001)
        # raw_input(k)


def main(test):

    # plot_data()

    # make_demo()

    # visualise_data()

    if test:
        check_s_w()
    else:
        reps()


if __name__=="__main__":

    # rospy.init_node('for_bullet_ctrlr')

    main(False)
