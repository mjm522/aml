import os
import time
import numpy as np
import matplotlib.pyplot as plt
from aml_rl_envs.utils.collect_demo import plot_demo, draw_trajectory
from aml_playground.peg_in_hole_reps.controller.sawyer_pih_reps import SawyerPegREPS
#get the experiment params
from aml_playground.peg_in_hole_reps.exp_params.experiment_params import exp_params


def main(joint_space=False):

    rewards = []

    ps = SawyerPegREPS(joint_space, exp_params)

    for i in range(5):

        print "Episode \t", i

        s = ps._eval_env.context()

        policy = ps._gpreps.run()

        _, mean_reward = ps._eval_env.execute_policy(policy.compute_w(s, transform=True, explore=False), s)

        print "mean_reward \t", mean_reward

        rewards.append(mean_reward)


    plt.figure("Mean reward")
    plt.plot(rewards)
    plt.show()

      
if __name__=="__main__":

    main()