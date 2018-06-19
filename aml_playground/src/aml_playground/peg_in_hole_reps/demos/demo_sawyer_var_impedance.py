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
from aml_playground.peg_in_hole_reps.controller.sawyer_var_imp_reps import SawyerVarImpREPS
#get the experiment params
from aml_rl_envs.utils.data_utils import save_csv_data
from aml_playground.peg_in_hole_reps.exp_params.experiment_var_imp_params import exp_params

def reps(joint_space=False):

    rewards = []
    params  = []
    force_penalties = []
    goal_penalties = []

    ps = SawyerVarImpREPS(joint_space, exp_params)

    plt.figure("Reward plots", figsize=(15,15))
    plt.ion()

    for i in range(100):

        print "Episode \t", i

        s = ps.context()

        policy = ps._gpreps.run()

        w = policy.compute_w(s, transform=True, explore=False)

        _, reward = ps.execute_policy(w)

        mean_reward = ps._penalty['total']
        force_penalty = ps._penalty['force']

        print "Parameter found*****************************************: \t", w
        print "mean_reward \t", mean_reward

        if mean_reward > -1000:
            rewards.append(mean_reward)

        force_penalties.append(force_penalty)

        goal_penalties.append(force_penalty+mean_reward)
        
        params.append(np.hstack([w,s,mean_reward]))

        ps._reset()
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

    file_name = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/reps_learned_params/params' +\
                                          time.strftime("_%b_%d_%Y_%H_%M_%S", time.localtime()) + '.csv'
    save_csv_data(file_name, np.asarray(params))
    raw_input("Press any key to exit")

def main(test):

    if test:
        check_s_w()
    else:
        reps()


if __name__=="__main__":

    rospy.init_node('sawyer_var_imp')
    main(False)