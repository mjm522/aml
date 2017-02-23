import numpy as np
import tensorflow as tf
from aml_io.tf_io import load_tf_check_point
from aml_dl.mdn.model.tf_model import tf_fwd_pushing_model


class NNPushForwardModel(object):

    def __init__(self, sess, network_params):

        self._sess = sess

        self._params = network_params

        self._device = self._params['device']

        with tf.device(self._device):
            self._net_ops = tf_fwd_pushing_model(dim_input= network_params['dim_input'],
                                                 dim_output = network_params['dim_output'],
                                                 n_hidden_layers = network_params['n_hidden_layers'], 
                                                 units_in_hidden_layers = network_params['units_in_hidden_layers'])

            self._init_op = tf.initialize_all_variables()

            self._saver = tf.train.Saver()


    def init_model(self):

        with tf.device(self._device):
            self._sess.run(self._init_op)

            if self._params['load_saved_model']:
                self.load_model()


    def load_model(self):

        load_tf_check_point(session=self._sess, filename=self._params['model_path'])

    def save_model(self):
        save_path = self._saver.save(self._sess, self._params['model_path'])
        print("Model saved in file: %s" % save_path)


    def train(self, x_data, y_data, epochs = 10000):
        with tf.device(self._device):
            # Keeping track of loss progress as we train
            loss = np.zeros(epochs) 

            train_op = self._net_ops['train']
            loss_op = self._net_ops['loss']

            for i in range(epochs):
              _, loss[i] = self._sess.run([train_op, loss_op],feed_dict={self._net_ops['x']: x_data, self._net_ops['y']: y_data})

            return loss


    def run_op(self, op_name, x_input):
        with tf.device(self._device):
            op = self._net_ops[op_name]

            out = self._sess.run(op, feed_dict={self._net_ops['x']: x_input})

            return out

    def sample_out(self, x_input, m_samples = 10):

        with tf.device(self._device):
            samples = self._sess.run([self._net_ops['output']], feed_dict={self._net_ops['x']: x_input})

            return samples


