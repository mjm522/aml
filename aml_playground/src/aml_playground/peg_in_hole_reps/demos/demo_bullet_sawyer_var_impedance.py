import os
import time
import rospy
import copy
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt
from aml_io.io_tools import save_data, load_data
from aml_rl_envs.sawyer.sawyer_var_imp_env import SawyerEnv
from aml_rl_envs.utils.collect_demo import plot_demo, draw_trajectory
from aml_playground.peg_in_hole_reps.controller.sawyer_var_imp_reps import SawyerVarImpREPS
#get the experiment params
from aml_rl_envs.utils.collect_demo import plot_demo
from aml_rl_envs.utils.data_utils import save_csv_data
from aml_playground.peg_in_hole_reps.exp_params.experiment_var_imp_params import exp_params



class DummyPolicy():
    """
    this is a dirty hack policy
    when a context is called it
    simply returns from a saved list of 
    values
    """

    def __init__(self, file_name):

        self._count = 0
        self._w_list = load_data(file_name)['w_list']

    def compute_w(self, context):
        w = self._w_list[self._count, :]
        self._count += 1
        return w


def test_params():

    file_name = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/reps_data.pkl'
    policy = DummyPolicy(file_name)
    
    env = SawyerEnv()
    env.execute_policy(policy=policy)
    
    raw_input("Press enter to exit")


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

        s_list, w_list, traj_data, reward = ps._eval_env.execute_policy(policy=policy,show_demo=False)

        mean_reward = ps._eval_env._penalty['total']
        force_penalty = ps._eval_env._penalty['force']

        print "Parameter found*****************************************: \t", np.mean(w_list, 0)
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

    w_list = np.asarray(w_list)

    force_traj = traj_data['ee_wrenches'][:,:3]
    plt.figure("Varying stiffness plot")
    plt.subplot(321)
    plt.title("Kx")
    plt.plot(w_list[:,0])
    plt.subplot(322)
    plt.title("Fx")
    plt.plot(force_traj[:,0])
    plt.subplot(323)
    plt.title("Ky")
    plt.plot(w_list[:,1])
    plt.subplot(324)
    plt.title("Fy")
    plt.plot(force_traj[:,1])
    plt.subplot(325)
    plt.title("Kz")
    plt.plot(w_list[:,2])
    plt.subplot(326)
    plt.title("Fz")
    plt.plot(force_traj[:,2])
    plt.show()


    data = {
    'w_list':w_list,
    's_list':np.asarray(s_list),
    'force_traj':force_traj,
    'ee_traj':traj_data['ee_traj'],
    'ee_vel_traj':traj_data['ee_vel_traj'],
    }

    file_name = os.environ['AML_DATA'] + '/aml_playground/imp_worlds/reps_data.pkl'
    save_data(data, file_name)
    raw_input("Press any key to exit")


def main():

    # reps()
    test_params()


if __name__ == '__main__':
    main()