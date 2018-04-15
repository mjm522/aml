import os
import time
import numpy as np
import matplotlib.pyplot as plt
from aml_lfd.utilities.smooth_demo_traj import SmoothDemoTraj
from aml_rl_envs.utils.collect_demo import plot_demo, draw_trajectory
from aml_playground.peg_in_hole_reps.controller.sawyer_pih_reps import SawyerPegREPS
#get the experiment params
from aml_rl_envs.utils.collect_demo import plot_demo
from aml_rl_envs.utils.data_utils import save_csv_data
from aml_playground.peg_in_hole_reps.exp_params.experiment_params import exp_params


def get_ee_traj(env, js_traj):

    ee_traj = env.fwd_simulate(js_traj, joint_space=True)

    file_name = os.environ['AML_DATA'] + '/aml_lfd/right_sawyer_exp_peg_in_hole/sawyer_bullet_ee_states.csv'

    dmt = SmoothDemoTraj(traj2smooth=ee_traj)

    plot_demo(ee_traj, color=[1,0,0], start_idx=0, life_time=0., cid=env._cid)

    # plot_demo(dmt._smoothed_traj, color=[0,0,1], start_idx=0, life_time=0., cid=env._cid)

    save_csv_data(file_name, dmt._smoothed_traj)

    raw_input()


def main(joint_space=False):

    rewards = []
    params  = []

    ps = SawyerPegREPS(joint_space, exp_params)

    while True:

        for k in range(2,3):
            ps.goto_hole(hole_id=k)
            ps.insert_hole(hole_id=k)
            ps._eval_env._sawyer.set_joint_state(ps._eval_env._sawyer._jnt_postns)

            raw_input("Press enter for next hole %d"%(k+2,))

    # get_ee_traj(ps._sim_env, ps._demo_traj)

    # plt.figure("Mean reward")
    # plt.figure("forces")
    # plt.ion()


    # plot_demo(dmp, start_idx=0, life_time=4)


    # forces1 = []
    # forces2 = []
    # forces3 = []


    # _,_,f,_ = ps._eval_env._sawyer.get_jnt_state(16)
    # print f#, t

    # forces1.append(f[0])
    # forces2.append(f[1])
    # forces3.append(f[2])



    # ee_pos, ee_ori = ps._eval_env._sawyer.get_ee_pose()
    # ee_traj.append(ee_pos)
    

    # for i in range(200):
    # plt.subplot(311)
    # plt.plot(forces1)

    # plt.subplot(312)
    # plt.plot(forces2)

    # plt.subplot(313)
    # plt.plot(forces3)

    #     # plt.pause(0.00001)
    # plt.show()

# def main(joint_space=False):

#     rewards = []
#     params  = []

#     ps = SawyerPegREPS(joint_space, exp_params)

#     # for k in range(5):

#     #     ps.goto_hole(hole_id=k)

#     # get_ee_traj(ps._sim_env, ps._demo_traj)

#     plt.figure("Mean reward")
#     plt.ion()

#     for i in range(200):

#         print "Episode \t", i

#         s = ps._eval_env.context()

#         policy = ps._gpreps.run()

#         w = policy.compute_w(s, transform=True, explore=False)

#         _, mean_reward = ps._eval_env.execute_policy(w, s, show_demo=False)

#         print "Parameter found*****************************************: \t", w
#         print "mean_reward \t", mean_reward

#         rewards.append(mean_reward)
#         params.append(np.hstack([w,s,mean_reward]))

#         ps._eval_env._reset()

#         plt.plot(rewards)
#         plt.pause(0.00001)
#         plt.draw()

#     file_name = os.environ['AML_DATA'] + '/aml_lfd/right_sawyer_exp_peg_in_hole/params.csv'
#     save_csv_data(file_name, np.asarray(params))
#     raw_input("Press any key to exit")

      
if __name__=="__main__":

    main()