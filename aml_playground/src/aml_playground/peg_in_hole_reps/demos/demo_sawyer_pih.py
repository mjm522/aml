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


# def main(joint_space=False):

#     rewards = []
#     params  = []

#     ps = SawyerPegREPS(joint_space, exp_params)

#     # j_pos = [-0.04216211,-0.33363672,-0.53082617,1.40670703,0.7603125,0.7803291,4.04997461]

#     # ps._eval_env._sawyer.set_joint_state(j_pos)

#     # k =0

#     while True:

#         ps._eval_env.simple_step()

#         # for k in range(2,3):
#         #     ps.goto_hole(hole_id=k)
#         #     ps.insert_hole(hole_id=k)

#         goal_pos, goal_ori = ps._eval_env._sawyer.get_ee_pose()
#         print goal_pos

#         # print ps._eval_env._sawyer.get_joint_state()[0]

#         # ps._eval_env._sawyer.set_joint_state(j_pos)#ps._eval_env._sawyer._jnt_postns)

#         # raw_input("Press enter for next hole %d"%(k+2,))

#     get_ee_traj(ps._sim_env, ps._demo_traj)

#     plt.figure("Mean reward")
#     plt.figure("forces")
#     plt.ion()


#     plot_demo(dmp, start_idx=0, life_time=4)


#     forces1 = []
#     forces2 = []
#     forces3 = []


#     _,_,f,_ = ps._eval_env._sawyer.get_jnt_state(16)
#     print f#, t

#     forces1.append(f[0])
#     forces2.append(f[1])
#     forces3.append(f[2])



#     ee_pos, ee_ori = ps._eval_env._sawyer.get_ee_pose()
#     ee_traj.append(ee_pos)
    


def main(joint_space=False):

    rewards = []
    params  = []

    ps = SawyerPegREPS(joint_space, exp_params)

    plt.figure("Mean reward")
    plt.ion()

    for i in range(70):

        print "Episode \t", i

        s = ps._eval_env.context()

        policy = ps._gpreps.run()

        w = policy.compute_w(s, transform=True, explore=False)

        _, mean_reward = ps._eval_env.execute_policy(w, s, show_demo=False)

        print "Parameter found*****************************************: \t", w
        print "mean_reward \t", mean_reward

        rewards.append(mean_reward)
        params.append(np.hstack([w,s,mean_reward]))

        ps._eval_env._reset()
        plt.clf()
        plt.plot(rewards, 'b')
        plt.pause(0.00001)
        plt.draw()

    file_name = os.environ['AML_DATA'] + '/aml_lfd/right_sawyer_exp_peg_in_hole/params.csv'
    save_csv_data(file_name, np.asarray(params))
    raw_input("Press any key to exit")

      
if __name__=="__main__":

    main()