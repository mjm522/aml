import os
import time
import rospy
import copy
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt
from aml_io.io_tools import save_data, load_data
from aml_rl_envs.utils.collect_demo import plot_demo, draw_trajectory
from aml_playground.peg_in_hole_reps.controller.sawyer_var_imp_reps import SawyerVarImpREPS
#get the experiment params
from aml_rl_envs.utils.collect_demo import plot_demo
from aml_rl_envs.utils.data_utils import save_csv_data
from aml_playground.peg_in_hole_reps.exp_params.experiment_var_imp_params import exp_params

def reps():

    rewards = []
    params  = []
    force_penalties = []
    goal_penalties = []

    ps = SawyerVarImpREPS(exp_params)

    plt.figure("Reward plots", figsize=(15,15))
    plt.ion()

    for i in range(100):

        print "Episode \t", i

        policy = ps._gpreps.run()

        _, reward = ps._eval_env.execute_policy(policy=policy,show_demo=False)

        mean_reward = ps._eval_env._penalty['total']
        force_penalty = ps._eval_env._penalty['force']

        print "Parameter found*****************************************: \t", w
        print "mean_reward \t", mean_reward

        if mean_reward > -1000:
            rewards.append(mean_reward)

        force_penalties.append(force_penalty)

        goal_penalties.append(force_penalty+mean_reward)
        
        params.append(np.hstack([mean_reward]))

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
    # save_csv_data(file_name, np.asarray(params))
    raw_input("Press any key to exit")


def main():

    reps()


if __name__ == '__main__':
    main()