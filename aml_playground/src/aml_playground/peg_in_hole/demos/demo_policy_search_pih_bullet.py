import os
import numpy as np
import pandas as pd
import tensorflow as tf
from gym.spaces import Box, Discrete
from aml_io.io_tools import save_data, load_data
from aml_robot.box2d.box2d_viewer import Box2DViewer
from aml_playground.peg_in_hole.policy_search.ddpg import DDPG
from aml_playground.peg_in_hole.policy_search.ou_noise import OUNoise
from aml_playground.peg_in_hole.pih_worlds.bullet.pih_world import PIHWorld
from aml_playground.peg_in_hole.pih_worlds.bullet.config import pih_world_config


def get_demo():
    """
    load the demo trajectory from the file
    """
    path_to_demo = pih_world_config['demo_folder_path'] + 'pih_ee_pos_data.csv'

    if not os.path.isfile(path_to_demo):
        raise Exception("The given path to demo does not exist, given path: \n" + path_to_demo)

    data_frame  = pd.DataFrame.from_csv(path_to_demo)

    demo_data  = np.asarray(data_frame['ee_position'])
    
    print demo_data

    raw_input()

    return demo_data

def view_traj(self, trajectory=get_demo()):
        """
        this funciton is specific for box2d viewer.
        """

        trajectory *=  viewer._config['pixels_per_meter']

        trajectory[:, 0] -=  viewer._config['cam_pos'][0]

        trajectory[:,1] =  viewer._config['image_height'] - viewer._config['cam_pos'][1] - trajectory[:,1]

        viewer._demo_point_list = trajectory.astype(int)



render = False
#specify parameters here:
episodes=10000

is_batch_norm = False #batch normalization switch

sess = tf.InteractiveSession()

env = PIHWorld(pih_world_config)


def main():

    reward_list = []

    traj2follow = get_demo()
    steps = traj2follow.shape[0]

    #add the trajectory to the viewer
    if render:
        view_traj(traj2follow)

    #Randomly initialize critic,actor,target critic, target actor network  and replay buffer
    exploration_noise = OUNoise(1) #env.action_space.shape[0]
    counter=0
    reward_per_episode = 0    
    total_reward=0
    num_states  = 6
    num_actions = 3 

    agent = DDPG(sess=sess, state_dim=num_states, action_dim=num_actions, 
                 action_max=[50., 50, 50], action_min=[1., 1., 1.], is_batch_norm=is_batch_norm)

    print "Number of States:", num_states
    print "Number of Actions:", num_actions
    print "Number of Steps per episode:", steps
    #saving reward:
    reward_st = np.array([0])
      
    
    for i in xrange(episodes):
        print "==== Starting episode no:",i,"====","\n"
        
        env.reset(noise=0.)

        #get data from the manipulator object
        jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque  = env._manipulator.get_jnt_state()

        #stack position and velocity
        observation = np.hstack([jnt_pos, jnt_vel])
        
        reward_per_episode = 0

        for t in xrange(steps-1):
            #rendering environmet (optional)            
            if render:
                viewer.draw()

            x = observation
            action = agent.evaluate_actor(np.reshape(x,[1,num_states]))
            noise  = exploration_noise.noise()
            action = action[0] + noise #Select action according to current policy and exploration noise
            
            print "Action at step", t ," :",action,"\n"
            
            # set_point = np.hstack([traj2follow[t, :], 0.1])
            set_point = traj2follow[t]

            ctrl_cmd = env.compute_os_ctrlr_cmd(os_set_point=set_point, Kp=action[0]) #20

            env.update(ctrl_cmd)

            for _ in range(pih_world_config['steps_per_frame']): 
                env.step()

            jnt_pos, jnt_vel, jnt_reaction_forces, jnt_applied_torque  = env._manipulator.get_jnt_state()

            observation = np.hstack([jnt_pos, jnt_vel])

            print type(traj2follow[t+1])

            raw_input()

            reward      = -0.01*np.linalg.norm(traj2follow[t+1] - jnt_vel)

            # if np.linalg.norm(data['j_pos'] - traj2follow[-1, :]) < 0.0

            if t == (steps-2):
                done = True
            else:
                done = False
            
            #add s_t,s_t+1,action,reward to experience memory
            agent.add_experience(observation_1=x,
                                 observation_2=observation,
                                 action=action,
                                 reward=reward,
                                 done=done)
            #train critic and actor network


            if counter > 64: 
                agent.train()

            reward_per_episode+=reward
            counter+=1

            #check if episode ends:
            if (done or (t == steps-1)):
                print 'EPISODE: ',i,' Steps: ',t,' Total Reward: ',reward_per_episode
                print "Printing reward to file"
                exploration_noise.reset() #reinitializing random noise for action exploration
                reward_st = np.append(reward_st,reward_per_episode)
                np.savetxt('episode_reward.txt',reward_st, newline="\n")
                print '\n\n'
                break


        print reward_per_episode

        # plt.plot(reward_list)
        # plt.pause(0.0001)
        # plt.draw()

    total_reward+=reward_per_episode            
    print "Average reward per episode {}".format(total_reward / episodes)    


if __name__ == '__main__':
    main()    