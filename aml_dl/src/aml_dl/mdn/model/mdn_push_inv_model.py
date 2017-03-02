
import tensorflow as tf
import numpy as np
from aml_io.tf_io import load_tf_check_point

from tf_mdn_model import MixtureDensityNetwork

class MDNPushInverseModel(object):


    def __init__(self, sess, network_params):

        self._sess = sess

        self._params = network_params

        self._device = self._params['device']

        with tf.device(self._device):


            self._mdn = MixtureDensityNetwork(network_params['dim_input'], 
                                              network_params['dim_output'], 
                                              network_params['k_mixtures'], 
                                              network_params['n_hidden'])

            self._mdn._init_model()
            self._net_ops = self._mdn._ops

            self._init_op = tf.global_variables_initializer()

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


    def sample_out(self, x_input, m_samples = 10):

        with tf.device(self._device):
            out_pi, out_mu, out_sigma = self._sess.run([self._net_ops['pi'], self._net_ops['mu'], self._net_ops['sigma']], feed_dict={self._net_ops['x']: x_input})


            samples = self._generate_mixture_samples(out_pi, out_mu, out_sigma,m_samples)

            return samples

    def sample_out_max_pi(self, x_input, m_samples = 10):

        with tf.device(self._device):
            out_pi, out_mu, out_sigma = self._sess.run([self._net_ops['pi'], self._net_ops['mu'], self._net_ops['sigma']], feed_dict={self._net_ops['x']: x_input})


            samples = self._generate_mixture_samples_from_max_pi(out_pi, out_mu, out_sigma,m_samples)

            return samples

    def expected_out(self, x_input, m_samples = 10):
        with tf.device(self._device):
            samples = self.sample_out(x_input,m_samples)[0]

            return np.mean(samples)


    def expected_out2(self, x_input, m_samples = 10):
        with tf.device(self._device):
            samples = self.sample_out(x_input,m_samples)[0]

            out = np.zeros(2)

            for i in range(m_samples):
                out += np.array([np.cos(samples[i]),np.sin(samples[i])])
            
            out /= m_samples

            out /= np.linalg.norm(out)

            return out

    def expected_max_pi_out(self, x_input, m_samples = 10):
        with tf.device(self._device):
            samples = self.sample_out_max_pi(x_input,m_samples)[0]

            return np.mean(samples)


    def expected_max_pi_out2(self, x_input, m_samples = 10):
        with tf.device(self._device):
            samples = self.sample_out_max_pi(x_input,m_samples)[0]

            out = np.zeros(2)

            for i in range(m_samples):
                out += np.array([np.cos(samples[i]),np.sin(samples[i])])
            
            out /= m_samples

            out /= np.linalg.norm(out)

            return out

    def run_op(self, op_name, x_input):
        with tf.device(self._device):
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

    def _max_pi_idx(self, pdf):
        
        i = np.argmax(pdf)

        return i

    def _sample_gaussian(self, rn, mu, std):

        return mu + rn*std


    def _generate_mixture_samples_from_max_pi(self, out_pi, out_mu, out_sigma, m_samples=10):

        # Number of test inputs
        N = out_mu.shape[0]

        M = m_samples

        result = np.random.rand(N, M)
        rn  = np.random.randn(N, M) # normal random matrix (0.0, 1.0)

        # Generates M samples from the mixture for each test input
        for j in range(M):
            for i in range(0, N):
              idx = self._max_pi_idx(out_pi[i])
              mu = out_mu[i, idx]
              std = out_sigma[i, idx]
              result[i, j] = self._sample_gaussian(rn[i, j], mu, std)

        return result


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



# def get_pi_idx(x, pdf):
#   N = pdf.size
#   accumulate = 0
#   for i in range(0, N):
#     accumulate += pdf[i]
#     if (accumulate >= x):
#       return i
#   print 'error with sampling ensemble'
#   return -1

# def generate_ensemble(out_pi, out_mu, out_sigma, M = 10):
#   NTEST = h_test.size
#   result = np.random.rand(NTEST, M) # initially random [0, 1]
#   rn = np.random.randn(NTEST, M) # normal random matrix (0.0, 1.0)
#   mu = 0
#   std = 0
#   idx = 0

#   # transforms result into random ensembles
#   for j in range(0, M):
#     for i in range(0, NTEST):
#       idx = get_pi_idx(result[i, j], out_pi[i])
#       mu = out_mu[i, idx]
#       std = out_sigma[i, idx]
#       result[i, j] = mu + rn[i, j]*std
#   return result




