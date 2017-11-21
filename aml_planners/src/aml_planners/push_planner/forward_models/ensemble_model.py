import os
import tensorflow as tf
import numpy as np
from aml_dl.mdn.model.tf_ensemble_mdn_model import EnsambleMDN



class EnsambleModel(object):


    def __init__(self, config, seed = 42):

        self._params = config['network_params']

        self._sess = tf.Session()

        tf.set_random_seed(config['random_seed'])

        self._ensamble_mdn = EnsambleMDN(self._params,self._sess)

        self._ensamble_mdn._init_model()



    def fit(self, X, y):

        loss = self._ensamble_mdn.train(X, y, self._sess, self._params['train_iterations'])

        print "Loss: ", loss


        return loss

    def predict(self, X):

        return self._ensamble_mdn.forward(self._sess, X, get_full_list=False)


