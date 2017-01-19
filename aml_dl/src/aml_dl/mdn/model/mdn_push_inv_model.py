
import tensorflow as tf
import numpy as np

from aml_dl.mdn.model.tf_model import tf_pushing_model
from aml_io.tf_io import load_tf_check_point

class MDNPushInverseModel(object):


    def __init__(self, sess, network_params):

        self._sess = sess

        self._params = network_params

        self._net_ops = tf_pushing_model(dim_input= network_params['dim_input'], 
                           n_hidden = network_params['n_hidden'], 
                           n_kernels = network_params['KMIX'])

        self._init_op = tf.initialize_all_variables()



    def init_model(self):

        self._sess.run(self._init_op)

        if self._params['load_saved_model']:
            self.load_model()


    def load_model(self):

        load_tf_check_point(session=self._sess, filename=self._params['model_path'])


    def train(self, x_data, y_data, epochs = 10000):

        # Keeping track of loss progress as we train
        loss = np.zeros(epochs) 

        train_op = self._net_ops['train']
        loss_op = self._net_ops['loss']

        for i in range(epochs):
          _, loss[i] = sess.run([train_op, loss_op],feed_dict={x: x_data, y: y_data})

        return loss


    def sample_out(self, x_input, m_samples = 10):

        out_pi, out_mu, out_sigma = self._sess.run([self._net_ops['pi'], self._net_ops['mu'], self._net_ops['sigma']], feed_dict={self._net_ops['x']: x_input})


        samples = self._generate_mixture_samples(out_pi, out_mu, out_sigma,m_samples)


        return samples

    def run_op(self, op_name, x_input):

        op = self._net_ops[op_name]

        out = self._sess.run(op, feed_dict={self._net_ops['x']: x_input})

        return out



    def _sample_pi_idx(self, x, pdf):
        N = pdf.size
        acc = 0
        for i in range(0, N):
            acc += pdf[i]
            if (acc >= x):
                return i

        print 'failed to sample mixture weight index'
        return -1

    def _sample_gaussian(self, rn, mu, std):

        return mu + rn*std


    def _generate_mixture_samples(self, out_pi, out_mu, out_sigma, m_samples=10):

        # Number of test inputs
        N = out_mu.shape[0]
        M = m_samples

        result = np.random.rand(N, M) # initially random [0, 1]
        rn  = np.random.randn(N, M) # normal random matrix (0.0, 1.0)
        mu  = 0
        std = 0
        idx = 0

        # Generates M samples from the mixture for each test input
        for j in range(M):
            for i in range(0, N):
              idx = self._sample_pi_idx(result[i, j], out_pi[i])
              mu = out_mu[i, idx]
              std = out_sigma[i, idx]
              result[i, j] = self._sample_gaussian(rn[i, j], mu, std)

        return result







