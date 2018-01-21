'''
reference https://github.com/stevenpjg/ddpg-aigym.git
'''

import math
import numpy as np
import tensorflow as tf

LEARNING_RATE = 0.0001
TAU = 0.001
BATCH_SIZE = 64
N_HIDDEN_1 = 400
N_HIDDEN_2 = 300

class ActorNet(object):
    """
    This is the function approximator class for the actor network
    """

    def __init__(self, sess, state_dim, action_dim, do_batch_norm=False):
        """
        Constructor for the actor networ
        Args:
        sess : valid tf session
        state_dim : dimensionality of the state
        action_dim : dimensionallity of the action
        do_batch_norm : should batch normalisation be performed
        """

        self._tf_sess = sess

        #creating actor network
        self._state    = tf.placeholder("float",[None, state_dim])    
        self._w1 = tf.Variable(tf.random_uniform([state_dim,N_HIDDEN_1],-1/math.sqrt(state_dim),1/math.sqrt(state_dim)))
        self._b1 = tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(state_dim),1/math.sqrt(state_dim)))
        self._w2 = tf.Variable(tf.random_uniform([N_HIDDEN_1,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1),1/math.sqrt(N_HIDDEN_1)))
        self._b2 = tf.Variable(tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1),1/math.sqrt(N_HIDDEN_1)))
        self._w3 = tf.Variable(tf.random_uniform([N_HIDDEN_2,action_dim],-0.003,0.003))
        self._b3 = tf.Variable(tf.random_uniform([action_dim],-0.003,0.003))
    
        hidden_layer_1 = tf.nn.softplus(tf.matmul(self._state, self._w1) + self._b1)
        
        if do_batch_norm:
            hidden_layer_1 = tf.layers.batch_normalization(hidden_layer_1)

        hidden_layer_2 = tf.nn.tanh(tf.matmul(hidden_layer_1, self._w2) + self._b2)

        if do_batch_norm:
            hidden_layer_2 = tf.layers.batch_normalization(hidden_layer_2)
        
        self._actor_model = tf.matmul(hidden_layer_2, self._w3) + self._b3

        #creating target actor network
        self._state_tgt    = tf.placeholder("float",[None, state_dim])    
        self._w1_tgt = tf.Variable(tf.random_uniform([state_dim,N_HIDDEN_1],-1/math.sqrt(state_dim),1/math.sqrt(state_dim)))
        self._b1_tgt = tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(state_dim),1/math.sqrt(state_dim)))
        self._w2_tgt = tf.Variable(tf.random_uniform([N_HIDDEN_1,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1),1/math.sqrt(N_HIDDEN_1)))
        self._b2_tgt = tf.Variable(tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1),1/math.sqrt(N_HIDDEN_1)))
        self._w3_tgt = tf.Variable(tf.random_uniform([N_HIDDEN_2,action_dim],-0.003,0.003))
        self._b3_tgt = tf.Variable(tf.random_uniform([action_dim],-0.003,0.003))


        hidden_layer_1_tgt = tf.nn.softplus(tf.matmul(self._state_tgt, self._w1_tgt) + self._b1_tgt)
        
        if do_batch_norm:
            hidden_layer_1_tgt = tf.layers.batch_normalization(hidden_layer_1_tgt)

        hidden_layer_2_tgt = tf.nn.tanh(tf.matmul(hidden_layer_1_tgt, self._w2_tgt) + self._b2_tgt)

        if do_batch_norm:
            hidden_layer_2_tgt = tf.layers.batch_normalization(hidden_layer_2_tgt)
        
        self._actor_model_tgt = tf.matmul(hidden_layer_2_tgt, self._w3_tgt) + self._b3_tgt


         #cost of actor network:
        self._q_gradient_input     = tf.placeholder("float",[None, action_dim]) #gets input from action_gradient computed in critic network file
        self._actor_parameters     = [self._w1, self._b1, self._w2, self._b2,self._w3, self._b3]
        self._parameters_gradients = tf.gradients(self._actor_model,self._actor_parameters,-self._q_gradient_input)#/BATCH_SIZE) changed -self.q_gradient to -
        
        self._optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,epsilon=1e-08).apply_gradients(zip(self._parameters_gradients,self._actor_parameters))  
        
        #initialize all tensor variable parameters:
        self._tf_sess.run(tf.initialize_all_variables())    
        
        #To make sure actor and target have same intial parmameters copy the parameters:
        # copy target parameters
        self._tf_sess.run([
            self._w1_tgt.assign(self._w1),
            self._b1_tgt.assign(self._b1),
            self._w2_tgt.assign(self._w2),
            self._b2_tgt.assign(self._b2),
            self._w3_tgt.assign(self._w3),
            self._b3_tgt.assign(self._b3)])

        #weighted average of target net and actor net
        self._update_target_actor_op = [
            self._w1_tgt.assign(TAU*self._w1 + (1-TAU)*self._w1_tgt),
            self._b1_tgt.assign(TAU*self._b1 + (1-TAU)*self._b1_tgt),  
            self._w2_tgt.assign(TAU*self._w2 + (1-TAU)*self._w2_tgt),
            self._b2_tgt.assign(TAU*self._b2 + (1-TAU)*self._b2_tgt),  
            self._w3_tgt.assign(TAU*self._w3 + (1-TAU)*self._w3_tgt),
            self._b3_tgt.assign(TAU*self._b3 + (1-TAU)*self._b3_tgt),]



    def evaluate_actor(self,state_t):
        """
        This function is to evaluate the actor
        """
        return self._tf_sess.run(self._actor_model, feed_dict={self._state:state_t})        
        
        
    def evaluate_target_actor(self, state_t_1):
        """
        This function is evaluate the target network
        """
        return self._tf_sess.run(self._actor_model_tgt, feed_dict={self._state_tgt: state_t_1})
        
    def train_actor(self,actor_state_in,q_gradient_input):
        """
        This function runs one loop of training of the network
        """
        self._tf_sess.run([self._optimizer], 
                                         feed_dict={ self._state: actor_state_in,
                                                     self._state_tgt: actor_state_in, 
                                                     self._q_gradient_input: q_gradient_input})
        
    def update_target_actor(self):
        """
        This fucntion updates the target actor network once in a while
        """
        self._tf_sess.run(self._update_target_actor_op)


    def get_params(self):
        """
        this function is to get all the variables of the network
        this will enable to save a snapshot of the parameters over the time
        """

        params = {
        'w1':self._w1,
        'w2':self._w2,
        'w3':self._w3,
        'b1':self._b1,
        'b2':self._b2,
        'b3':self._b3,
        'w1_tgt':self._w1_tgt,
        'w2_tgt':self._w2_tgt,
        'w3_tgt':self._w3_tgt,
        'b1_tgt':self._b1_tgt,
        'b2_tgt':self._b2_tgt,
        'b3_tgt':self._b3_tgt,
        }

        return params
