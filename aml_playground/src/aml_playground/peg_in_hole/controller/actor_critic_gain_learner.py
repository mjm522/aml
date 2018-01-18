import collections
import numpy as np
import tensorflow as tf

from aml_robot.box2d.box2d_viewer import Box2DViewer
from aml_playground.peg_in_hole.pih_worlds.box2d.config import pih_world_config
from aml_playground.peg_in_hole.pih_worlds.box2d.box2d_pih_world import Box2DPIHWorld

np.random.seed(123)

class ActorCriticGainLearner(object):

    """
    This class interfaces the actor and critic network
    At present this code is written just as an interface to
    Box2d object, which can be changed easily
    """

    def __init__(self, traj2follow, discount_factor=1.0, num_episodes=50):

        """
        Constructor of the actor critic network
        """

        #trajectory to be followed
        self._traj2follow  = traj2follow

        #world
        self._env          = Box2DPIHWorld(pih_world_config)

        #visualizer
        self._viewer       = Box2DViewer(self._env, pih_world_config, is_thread_loop=False)

        #add the trajectory to the viewer
        self.view_traj(self._traj2follow)

        #num steps
        self._time_steps   = self._traj2follow.shape[0]

        #discount factor for the reward calculation
        self._discount_factor = discount_factor

        self._num_episodes    = num_episodes

        #stats
        self._episode_lengths = np.zeros(self._num_episodes)
        self._episode_rewards = np.zeros(self._num_episodes)

    def view_traj(self, trajectory):
        """
        this funciton is specific for box2d viewer.
        """

        trajectory *= self._viewer._config['pixels_per_meter']

        trajectory[:, 0] -= self._viewer._config['cam_pos'][0]

        trajectory[:,1] = self._viewer._config['image_height'] - self._viewer._config['cam_pos'][1] - trajectory[:,1]

        self._viewer._demo_point_list = trajectory.astype(int)


    def setup(self, sess):
        """
        create the actor critic network
        Args: sess, tf session started
        """

        #tf session
        self._tf_sess = sess

        #Policy Function to be optimized 
        self._actor_net    = ActorNetwork(sess=self._tf_sess, 
                                          state_dim=6, 
                                          action_dim=1, 
                                          learning_rate=0.001, 
                                          action_space_low=1., 
                                          action_space_high=50.,)

        #Value function approximator, used as a critic
        self._critic_net   = CriticNetwork(sess=self._tf_sess, 
                                           state_dim=6, 
                                           learning_rate=0.1)


    def run(self, train=True, render=False):

        """
        main code that trains the network
        """

        #a named tuple to collect data
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state"])

        #start the iteration
        for i_episode in range(self._num_episodes):
            # Reset the environment and pick the fisrst action
            #reseting to the first intial state without noise
            state = self._env.reset(noise=0.)

            #get data from the manipulator object
            data  = self._env._manipulator.get_state()

            #stack position and velocity
            state = np.hstack([data['j_pos'], data['j_vel']])
            
            episode = []
            
            # One step in the environment
            for t in range(self._time_steps-1):
                
                # Take a step
                Kp = self._actor_net.predict(state)

                print "Kp value predicted \t", Kp[0]

                set_point = np.hstack([self._traj2follow[t, :], 0.1])

                action = self._env._manipulator.compute_os_ctrlr_cmd(os_set_point=set_point, Kp=Kp) #20

                self._env.update(action)

                for i in range(self._viewer._steps_per_frame): 
                    self._env.step()

                if render:
                    self._viewer.draw()

                data       = self._env._manipulator.get_state()

                next_state = np.hstack([data['j_pos'], data['j_vel']])

                reward     = -0.01*np.linalg.norm(np.hstack([self._traj2follow[t+1, :], 0.1]) - data['j_pos'])

                if train:
                
                    # Keep track of the transition
                    episode.append(Transition(state=state, action=Kp, reward=reward, next_state=next_state))
                    
                    # Update statistics
                    self._episode_rewards[i_episode] += reward
                    self._episode_lengths[i_episode] = t
                    
                    # Calculate TD Target
                    value_nexttf.contrib.distributions.Normal = self._critic_net.predict(next_state)
                    td_target = reward + self._discount_factor * value_next
                    #advantage computation
                    td_error = td_target - self._critic_net.predict(state)
                    
                    # Update the value estimator
                    self._critic_net.update(state, td_target)
                    
                    # Update the policy estimator
                    # using the td error as our advantage estimate
                    self._actor_net.update(state, td_error, Kp)
  
                    # Print out which step we're on, useful for debugging.
                    print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, self._num_episodes, self._episode_rewards[i_episode - 1]))

                else:

                    print "Step \t", t

                state = next_state

            #execute the loop only once
            #this is to see the environment alone
            #working
            if not train:

                break
        
        return self._episode_lengths, self._episode_rewards



class ActorNetwork(object):
    """
    Policy Function approximator. 
    """
    
    def __init__(self, sess, state_dim, action_dim, action_space_low, action_space_high, learning_rate=0.01, scope="policy_estimator"):
        
        """
        Constructor of the actor critic network
        """
        self._tf_sess = sess

        with tf.variable_scope(scope):
            self._state  = tf.placeholder(tf.float32, [state_dim], "state")
            self._target= tf.placeholder(dtype=tf.float32, name="target")

            # This is just linear classifier
            self._mu = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self._state, 0),
                num_outputs=action_dim,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)
            self._mu = tf.squeeze(self._mu)
            
            self._sigma = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self._state, 0),
                num_outputs=action_dim,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)
            
            self._sigma = tf.squeeze(self._sigma)
            self._sigma = tf.nn.softplus(self._sigma) + 1e-5
            self._normal_dist = tf.contrib.distributions.Normal(self._mu, self._sigma)
            self._action = self._normal_dist._sample_n(1)
            self._action = tf.clip_by_value(self._action, action_space_low, action_space_high)

            # Loss and train op
            self._loss = -self._normal_dist.log_prob(self._action) * self._target
            # Add cross entropy cost to encourage exploration
            self._loss -= 1e-1 * self._normal_dist.entropy()
            
            self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self._train_op = self._optimizer.minimize(self._loss, global_step=tf.contrib.framework.get_global_step())
    
    def predict(self, state, sess=None):
        """
        compute the action by forward pass through network

        Args: state : input state of the system. Should be of same dimentions as self._state
        """
        sess = sess or self._tf_sess
        
        return sess.run(self._action, { self._state: state })

    def update(self, state, target, action, sess=None):
        """
        train the network
        """
        sess = sess or self._tf_sess
        #create the feed variables
        feed_dict = { self._state: state, self._target: target, self._action: action }
        #train the network
        _, loss = sess.run([self._train_op, self._loss], feed_dict)

        return loss


class CriticNetwork(object):
    """
    Value Function approximator. 
    """
    
    def __init__(self, sess, state_dim, learning_rate=0.1, scope="value_function_estimator"):

        """
        Constructor of the actor critic network
        """

        self._tf_sess = sess

        with tf.variable_scope(scope):
            self._state  = tf.placeholder(tf.float32, [state_dim], "state")
            self._target= tf.placeholder(dtype=tf.float32, name="target")

            # This is just linear classifier
            self._output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self._state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self._value_estimate = tf.squeeze(self._output_layer)
            self._loss = tf.squared_difference(self._value_estimate, self._target)

            self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self._train_op = self._optimizer.minimize(self._loss, global_step=tf.contrib.framework.get_global_step())        
    
    def predict(self, state, sess=None):
        """
        compute the action by forward pass through network

        Args: state : input state of the system. Should be of same dimentions as self._state
        """
        sess = sess or self._tf_sess
        return sess.run(self._value_estimate, { self._state: state })

    def update(self, state, target, sess=None):
        """
        train the network
        """
        sess = sess or self._tf_sess
        #create the feed variables
        feed_dict = { self._state: state, self._target: target }
        #train the network
        _, loss = sess.run([self._train_op, self._loss], feed_dict)
        return loss



