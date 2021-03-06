import os
import numpy as np
import pybullet as pb
import tensorflow as tf
from gym.spaces import Box, Discrete
from aml_io.io_tools import save_data, load_data
from aml_robot.box2d.box2d_viewer import Box2DViewer
from aml_visual_tools.plot_data_stream import PlotDataStream
from aml_playground.peg_in_hole.policy_search.ddpg import DDPG
from aml_playground.peg_in_hole.policy_search.ou_noise import OUNoise
from aml_playground.peg_in_hole.pih_worlds.bullet.pih_world import PIHWorld
from aml_playground.peg_in_hole.pih_worlds.bullet.config import pih_world_config

np.random.seed(123)
tf.set_random_seed(123)

render = True
#specify parameters here:
episodes=10000

show_reward_traj = True

is_batch_norm = False #batch normalization switch

sess = tf.InteractiveSession()

env = PIHWorld(pih_world_config)

# reward_plotter = PlotDataStream(plot_title="reward_plot", plot_size=None, max_plot_length=20000)
loss_plotter  = PlotDataStream(plot_title="loss_plot", plot_size=None, max_plot_length=20000)

def get_demo():
    """
    load the demo trajectory from the file
    """
    path_to_demo = pih_world_config['demo_folder_path'] + 'pih_js_ee_pos_data.csv'

    if not os.path.isfile(path_to_demo):
        raise Exception("The given path to demo does not exist, given path: \n" + path_to_demo)

    demo_data  = np.genfromtxt(path_to_demo, delimiter=',')

    return demo_data

def view_traj(self, trajectory=get_demo()):
        """
        this funciton is to load trajectory into the bullet viewer.
        the state is a list of 6 values, only the x,y,z values are taken
        """
        for k in range(len(trajectory)-1):

            env.draw_trajectory(point_1=trajectory[k, 6:9], point_2=trajectory[k+1, 6:9], colour=[0,0,1], line_width=5.5)


def update_reward_plot(reward):
    reward_plotter.add_data(reward)
    reward_plotter.update_plot()

def update_loss_plot(loss):
    loss_plotter.add_data(loss)
    loss_plotter.update_plot()


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
    num_states  = 12
    num_actions = 3 

    agent = DDPG(sess=sess, state_dim=num_states, action_dim=num_actions, 
                 action_max=[46.1, 46.1, 46.1], action_min=[0., 0., 0.], 
                 is_batch_norm=is_batch_norm,
                 snapshot_path=pih_world_config['train_data_storage_path'],
                 restore_model= True)

    print "Number of States:", num_states
    print "Number of Actions:", num_actions
    print "Number of Steps per episode:", steps
    #saving reward:
    reward_st = np.array([0])
    
    loss_list_episode = []
    
    for i in xrange(episodes):
        print "==== Starting episode no:",i,"====","\n"
        
        env.reset(noise=0.)

        #get data from the manipulator object
        observation  = env.get_observation()
        
        reward_per_episode = 0

        loss_list = []

        for t in xrange(steps-1):

            done = False
    
            x = observation
            action = agent.evaluate_actor(np.reshape(x,[1,num_states]))
            noise  = np.abs(exploration_noise.noise())

            action = action[0] + noise #Select action according to current policy and exploration noise

            # print "Action is \t", action
            
            # print "Action at step", t ," :",action,"\n"
    
            #position setpoint
            set_point = traj2follow[t, :3]
            set_point_vel = traj2follow[t, 3:6]

            error = np.linalg.norm(set_point - observation[6:9])

            # while error > 0.05:

                # ctrl_cmd = env.compute_os_ctrlr_cmd(os_set_point=set_point, Kp=action[0]) #20
                # ctrl_cmd = env.compute_os_imp_ctrlr_cmd(os_set_point=set_point, Kp=action[0])
            # ctrl_cmd = env.compute_js_ctrlr_cmd(js_set_point=set_point, Kp=action[0])

            pb.setJointMotorControlArray(env._robot_id, 
                             env._manipulator._joint_idx, 
                             controlMode=pb.POSITION_CONTROL,
                             targetPositions=set_point,
                             targetVelocities=set_point_vel,
                             positionGains=action.tolist(),
                             velocityGains=np.sqrt(action).tolist())

            # env.update(ctrl_cmd)

                # error = np.linalg.norm(traj2follow[t+1, :3] - observation[:3])

            env.step()

            observation = env.get_observation()

                # print "step \t", t, "error \t", error

            reward      = -0.01*np.linalg.norm(traj2follow[t+1, :3] - observation[:3])

            #the done is only done if the agent was able to reach the specified 
            #done after right number of steps
            if t == (steps-2):

                error = np.linalg.norm(traj2follow[-1, :3] - observation[:3])

                if error < 0.01:
                    done = True
                    reward = 10.
                    print "Wow... done!"
                else:
                    print "Final error \t", error
                    reward = -10.

       
            #add s_t,s_t+1,action,reward to experience memory
            agent.add_experience(observation_1=x,
                                 observation_2=observation,
                                 action=action,
                                 reward=reward,
                                 done=done)
            #train critic and actor network


            if counter > 64: 
                loss = agent.train()

                loss_list.append(loss)

            reward_per_episode+=reward
            counter+=1

            #check if episode ends:
            if (done or (t == steps-2)):
                print 'EPISODE: ',i,' Steps: ',t,' Total Reward: ',reward_per_episode
                print "Printing reward to file"
                exploration_noise.reset() #reinitializing random noise for action exploration
                reward_st = np.append(reward_st,reward_per_episode)
                np.savetxt('episode_reward.txt',reward_st, newline="\n")
                print '\n\n'
                break

        if show_reward_traj:

            if loss_list:
                update_loss_plot(np.sum(np.asarray(loss_list)))

            # update_reward_plot(reward_per_episode)

        #save the parameters of the network every
        #100th episode
        if (i%100 == 0) and (i > 0):
            agent.save_snapshot(epi_num=i)

    total_reward+=reward_per_episode            
    print "Average reward per episode {}".format(total_reward / episodes)    


if __name__ == '__main__':
    main()    