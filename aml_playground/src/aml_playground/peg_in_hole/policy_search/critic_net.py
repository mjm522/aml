'''
reference https://github.com/stevenpjg/ddpg-aigym.git
'''
import math
import numpy as np
import tensorflow as tf

LEARNING_RATE = 0.001
TAU = 0.001
BATCH_SIZE = 64
N_HIDDEN_1 = 400
N_HIDDEN_2 = 300

class CriticNet(object):
    """
    This is the function approximator class for the critic network
    """

    def __init__(self, sess, state_dim, action_dim, do_batch_norm=False):
        """
        Constructor for the critic networ
        Args:
        sess : valid tf session
        state_dim : dimensionality of the state
        action_dim : dimensionallity of the action
        do_batch_norm : should batch normalisation be performed
        """

        self._tf_sess = sess

        #creating critic network
        self._state    = tf.placeholder("float",[None, state_dim])
        self._action   = tf.placeholder("float",[None, action_dim])

        self._w1 = tf.Variable(tf.random_uniform([state_dim,N_HIDDEN_1],-1/math.sqrt(state_dim),1/math.sqrt(state_dim)))
        self._b1 = tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(state_dim),1/math.sqrt(state_dim)))
        self._w2 = tf.Variable(tf.random_uniform([N_HIDDEN_1,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1),1/math.sqrt(N_HIDDEN_1)))
        self._b2 = tf.Variable(tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1),1/math.sqrt(N_HIDDEN_1)))
        self._w3 = tf.Variable(tf.random_uniform([N_HIDDEN_2,action_dim],-0.003,0.003))
        self._b3 = tf.Variable(tf.random_uniform([action_dim],-0.003,0.003))

        self._w2_action = tf.Variable(tf.random_uniform([action_dim,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+action_dim),1/math.sqrt(N_HIDDEN_1+action_dim)))

    
        hidden_layer_1 = tf.matmul(self._state, self._w1)
        
        if do_batch_norm:
            hidden_layer_1 = tf.layers.batch_normalization(hidden_layer_1)

        hidden_layer_1 = tf.nn.softplus(hidden_layer_1) + self._b1

        hidden_layer_2 = tf.matmul(hidden_layer_1, self._w2)+tf.matmul(self._action, self._w2_action)

        if do_batch_norm:
            hidden_layer_2 = tf.layers.batch_normalization(hidden_layer_2)


        hidden_layer_2 = tf.nn.softplus(hidden_layer_2) + self._b2

        
        self._critic_model = tf.matmul(hidden_layer_2, self._w3) + self._b3
        

        #creating target critic network
        self._state_tgt    = tf.placeholder("float",[None, state_dim])
        self._action_tgt   = tf.placeholder("float",[None, action_dim])
        

        self._w1_tgt = tf.Variable(tf.random_uniform([state_dim,N_HIDDEN_1],-1/math.sqrt(state_dim),1/math.sqrt(state_dim)))
        self._b1_tgt = tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(state_dim),1/math.sqrt(state_dim)))
        self._w2_tgt = tf.Variable(tf.random_uniform([N_HIDDEN_1,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1),1/math.sqrt(N_HIDDEN_1)))
        self._b2_tgt = tf.Variable(tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1),1/math.sqrt(N_HIDDEN_1)))
        self._w3_tgt = tf.Variable(tf.random_uniform([N_HIDDEN_2,action_dim],-0.003,0.003))
        self._b3_tgt = tf.Variable(tf.random_uniform([action_dim],-0.003,0.003))

        self._w2_action_tgt = tf.Variable(tf.random_uniform([action_dim,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+action_dim),1/math.sqrt(N_HIDDEN_1+action_dim)))


        hidden_layer_1_tgt = tf.matmul(self._state_tgt, self._w1_tgt)

        if do_batch_norm:
            hidden_layer_1_tgt = tf.layers.batch_normalization(hidden_layer_1_tgt)

        hidden_layer_1_tgt = tf.nn.softplus(hidden_layer_1_tgt) + self._b1_tgt

        hidden_layer_2_tgt = tf.matmul(hidden_layer_1_tgt, self._w2_tgt) + tf.matmul(self._action_tgt, self._w2_action_tgt)

        if do_batch_norm:
            hidden_layer_2_tgt = tf.layers.batch_normalization(hidden_layer_2_tgt)

        hidden_layer_2_tgt = tf.nn.tanh(hidden_layer_2_tgt) + self._b2_tgt
         
        self._critic_model_tgt = tf.matmul(hidden_layer_2_tgt, self._w3_tgt) + self._b3_tgt

        #input q value
        self._q_value = tf.placeholder("float",[None,1]) #supervisor
        #self.l2_regularizer_loss = tf.nn.l2_loss(self.W1_c)+tf.nn.l2_loss(self.W2_c)+ tf.nn.l2_loss(self.W2_action_c) + tf.nn.l2_loss(self.W3_c)+tf.nn.l2_loss(self.B1_c)+tf.nn.l2_loss(self.B2_c)+tf.nn.l2_loss(self.B3_c) 
        l2_regularizer_loss = 0.0001*tf.reduce_sum(tf.pow(self._w2, 2))             
        
        self._cost = tf.pow(self._critic_model - self._q_value, 2)/BATCH_SIZE + l2_regularizer_loss #/tf.to_float(tf.shape(self.q_value_in)[0])
        
        self._optimizer  = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self._cost)
        self._act_grad_v = tf.gradients(self._critic_model, self._action)

        self._action_gradients = [self._act_grad_v[0]/tf.to_float(tf.shape(self._act_grad_v[0])[0])] #this is just divided by batch size
        
        #from simple actor net:
        self._check_fl = self._action_gradients             
        
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


    def train_critic(self, state_t_batch, action_batch, y_i_batch ):

        self._tf_sess.run([self._optimizer],\
                                            feed_dict={self._state: state_t_batch,
                                                       self._state_tgt: state_t_batch, 
                                                       self._action:action_batch, 
                                                       self._action_tgt:action_batch, 
                                                       self._q_value: y_i_batch})
        
    def evaluate_target_critic(self, state_t_1, action_t_1):

        return self._tf_sess.run(self._critic_model_tgt,\
                                                  feed_dict={self._state_tgt: state_t_1, 
                                                  self._action_tgt: action_t_1})    
        
        
    def compute_delQ_a(self, state_t, action_t):

        return self._tf_sess.run(self._action_gradients,\
                                                        feed_dict={self._state: state_t,
                                                                   self._action: action_t})

    def update_target_critic(self):

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
        'w2_action':self._w2_action,
        'w2_action_tgt':self._w2_action_tgt,
        }

        return params

