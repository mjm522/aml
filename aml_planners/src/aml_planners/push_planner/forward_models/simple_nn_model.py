import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model

class SimpleNNModel(object):

    def __init__(self, config):

        self._params = config['network_params']

        self._model = None

    def create_model(self):
        # create model
        self._model = Sequential()
        self._model.add(Dense(30, input_dim=self._params['state_dim']+self._params['cmd_dim']))
        self._model.add(Activation("relu"))
        self._model.add(Dense(30, ))
        self._model.add(Activation("relu"))
        self._model.add(Dense(30, ))
        self._model.add(Activation("relu"))
        self._model.add(Dense(30, ))
        self._model.add(Activation("relu"))
        self._model.add(Dense(30, ))
        self._model.add(Activation("relu"))
        self._model.add(Dense(30, ))
        self._model.add(Activation("relu"))
        self._model.add(Dense(self._params['out_dim'])) 
        self._model.add(Activation("linear"))
        self._model.compile(loss='mean_absolute_error', optimizer='adam', metrics=["accuracy"])

    def plot(self, Y_train, predictions):

        plt.figure(2)
        plt.subplot(311)
        plt.plot(Y_train[:,0], 'r')
        plt.plot(predictions[:,0], 'g')

        plt.subplot(312)
        plt.plot(Y_train[:,1], 'r')
        plt.plot(predictions[:,1], 'g')

        plt.subplot(313)
        plt.plot(Y_train[:,2], 'r')
        plt.plot(predictions[:,2], 'g')

        plt.show()

    def fit(self, X=None, y=None):

        if self._params['load_saved_model']:
            self._model = load_model(self._params['model_path'])
        else:
            self.create_model()
            history = self._model.fit(X, y, epochs=self._params['epochs'], batch_size=self._params['batch_size'])
            if self._params['save_model']:
                self._model.save(self._params['model_path'])
            return history.history['loss']
        
    def predict(self, X):

        X = np.r_[X[0][:2], X[0][-1]][None,:]

        prediction = self._model.predict(X)

        sigmas =  np.zeros(X.shape[0])

        prediction = np.r_[prediction[0][-1]*prediction[0][:2], 0.,0.,0.,0.]

        return prediction, sigmas
