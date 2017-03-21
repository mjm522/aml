import os
import numpy as np
import tensorflow as tf
from aml_dl.mdn.model.tf_models import tf_siamese_model

class SiamesePushModel(object):
    
    def __init__(self, sess, network_params):

        self._sess = sess

        self._params = network_params

        self._device = self._params['device']

        self._tf_sumry_wrtr = None

        self._optimiser = network_params['optimiser']

        self._data_configured = False

        if network_params['write_summary']:
            self._tf_sumry_wrtr = TfSummaryWriter(sess)
            cuda_path = '/usr/local/cuda/extras/CUPTI/lib64'

            curr_ld_path = os.environ["LD_LIBRARY_PATH"]

            if not cuda_path in curr_ld_path.split(os.pathsep):
                print "Enviroment variable LD_LIBRARY_PATH does not contain %s"%cuda_path
                print "Please add it, else the program will crash!"
                raw_input("Press Ctrl+C")
                # os.environ["LD_LIBRARY_PATH"] = curr_ld_path + ':'+cuda_path

        with tf.device(self._device):
            self._net_ops = tf_siamese_model(dim_output=network_params['dim_output'],
                                     loss_type='normal',
                                     cnn_params=network_params['cnn_params'], 
                                     fc_params=network_params['fc_params'],
                                     optimiser_params=network_params['optimiser'],
                                     tf_sumry_wrtr=self._tf_sumry_wrtr)

            self._init_op = tf.initialize_all_variables()

            self._saver = tf.train.Saver()

    def init_model(self):

        with tf.device(self._device):
            self._sess.run(self._init_op)

            if self._params['load_saved_model']:
                self.load_model()