'''
reference https://github.com/stevenpjg/ddpg-aigym.git
'''

import random
import numpy as np
import tensorflow as tf
from collections import deque
from actor_net import ActorNet
from critic_net import CriticNet
from gym.spaces import Box, Discrete
from aml_io.io_tools import save_data

from tensorflow_grad_inverter import grad_inverter

REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA=0.99
is_grad_inverter = True

class DDPG:
    
    """ Deep Deterministic Policy Gradient Algorithm"""
    def __init__(self, sess, state_dim, action_dim, action_max, action_min, is_batch_norm, snapshot_path=None, restore_model=False):
        self._tf_sess = sess

        self._num_states  = state_dim  
        self._num_actions = action_dim 
        
        self._critic_net = CriticNet(sess, self._num_states, self._num_actions) 
        self._actor_net  = ActorNet(sess,  self._num_states, self._num_actions)
            
        #Initialize Buffer Network:
        self._replay_memory = deque()
        
        #Intialize time step:
        self._time_step = 0
        self._counter = 0
        
        self._action_max = action_max 
        self._action_min = action_min         
        action_bounds = [self._action_max, self._action_min] 
        self._grad_inv = grad_inverter(sess, action_bounds)


        # Add ops to save and restore all the variables.
        self._saver = tf.train.Saver()

        if snapshot_path is None:
            #if no paths are given, store in the current folder
            self._snapshot_path = './'
        else:
            self._snapshot_path = snapshot_path

        if restore_model:
            all_ckpt = tf.train.get_checkpoint_state(self._snapshot_path, 'checkpoint').all_model_checkpoint_paths
            self._saver.restore(sess, all_ckpt[-1])
            print "Successfully loaded the model..."
        
        
    def evaluate_actor(self, state_t):
        action_list_tmp = self._actor_net.evaluate_actor(state_t)
        action_list = []

        #enforce the bounds on the actions
        for action in action_list_tmp:
            action_list.append(np.maximum( np.minimum(action, self._action_max), self._action_min ) )

        return action_list
    
    def add_experience(self, observation_1, observation_2, action, reward, done):
        self._observation_1 = observation_1
        self._observation_2 = observation_2
        self._action = action
        self._reward = reward
        self._done = done
        self._replay_memory.append((self._observation_1, self._observation_2, self._action, self._reward,self._done))
        self._time_step = self._time_step + 1
        if(len(self._replay_memory)>REPLAY_MEMORY_SIZE):
            self._replay_memory.popleft()
            
        
    def minibatches(self):
        batch = random.sample(self._replay_memory, BATCH_SIZE)
        #state t
        self._state_t_batch = [item[0] for item in batch]
        self._state_t_batch = np.array(self._state_t_batch)
        #state t+1        
        self._state_t_1_batch = [item[1] for item in batch]
        self._state_t_1_batch = np.array( self._state_t_1_batch)
        self._action_batch = [item[2] for item in batch]
        self._action_batch = np.array(self._action_batch)
        self._action_batch = np.reshape(self._action_batch,[len(self._action_batch),self._num_actions])
        self._reward_batch = [item[3] for item in batch]
        self._reward_batch = np.array(self._reward_batch)
        self._done_batch = [item[4] for item in batch]
        self._done_batch = np.array(self._done_batch)  
                  
                 
    def train(self):
        #sample a random minibatch of N transitions from R
        self.minibatches()
        self._action_t_1_batch = self._actor_net.evaluate_target_actor(self._state_t_1_batch)
        #Q'(s_i+1,a_i+1)        
        q_t_1 = self._critic_net.evaluate_target_critic(self._state_t_1_batch, self._action_t_1_batch) 
        self._y_i_batch=[]         
        for i in range(0,BATCH_SIZE):
                           
            if self._done_batch[i]:
                self._y_i_batch.append(self._reward_batch[i])
            else:
                
                self._y_i_batch.append(self._reward_batch[i] + GAMMA*q_t_1[i][0])                 
        
        self._y_i_batch=np.array(self._y_i_batch)
        self._y_i_batch = np.reshape(self._y_i_batch,[len(self._y_i_batch),1])
        
        # Update critic by minimizing the loss
        self._critic_net.train_critic(self._state_t_batch, self._action_batch,self._y_i_batch)
        
        # Update actor proportional to the gradients:
        action_for_delQ = self.evaluate_actor(self._state_t_batch) 
        
        if is_grad_inverter:        
            self._del_Q_a = self._critic_net.compute_delQ_a(self._state_t_batch,action_for_delQ)#/BATCH_SIZE            
            self._del_Q_a = self._grad_inv.invert(self._del_Q_a,action_for_delQ) 
        else:
            self._del_Q_a = self._critic_net.compute_delQ_a(self._state_t_batch,action_for_delQ)[0]#/BATCH_SIZE
        
        # train actor network proportional to delQ/dela and del_Actor_model/del_actor_parameters:
        self._actor_net.train_actor(self._state_t_batch,self._del_Q_a)
 
        # Update target Critic and actor network
        self._critic_net.update_target_critic()
        self._actor_net.update_target_actor()

    def save_snapshot(self, epi_num):
        """
        This function is to save parameters of both actor net and critic net 
        to a ckpt file
        """

        filename = self._snapshot_path +'episode_' + str(epi_num) + 'model.ckpt'

        self._saver.save(self._tf_sess, filename, global_step=epi_num, write_meta_graph=False)

        print "Snapshot saved for episode \t", epi_num


