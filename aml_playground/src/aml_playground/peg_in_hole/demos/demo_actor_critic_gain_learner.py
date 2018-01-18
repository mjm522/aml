import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from aml_io.io_tools import save_data, load_data
from aml_playground.peg_in_hole.controller.actor_critic_gain_learner import ActorCriticGainLearner


def get_demo():
    """
    load the demo trajectory from the file
    """
    data_storage_path = os.environ['AML_DATA'] + '/aml_playground/pih_worlds/box2d/demos/' 
    path_to_demo = data_storage_path + 'demo.pkl'

    if not os.path.isfile(path_to_demo):
        raise Exception("The given path to demo does not exist, given path: \n" + path_to_demo)

    trajectory = np.asarray(load_data(path_to_demo))

    return trajectory


def plot_episode_stats(episode_lengths, episode_rewards, smoothing_window=10):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(episode_lengths), np.arange(len(episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")

    plt.show()


tf.reset_default_graph()

acgl = ActorCriticGainLearner(traj2follow=get_demo(), discount_factor=0.95)

with tf.device("/cpu:0"):

    with tf.Session() as sess:

        acgl.setup(sess=sess)

        sess.run(tf.initialize_all_variables())

        print "Training ..."

        episode_lengths, episode_rewards = acgl.run(train=True, render=False)


plot_episode_stats(episode_lengths, episode_rewards)