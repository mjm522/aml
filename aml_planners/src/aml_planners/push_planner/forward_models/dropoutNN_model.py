import sys
import pylab
import random
import numpy as np
np.random.seed(0)


import keras.backend as K
from keras import initializers
from keras.engine import InputSpec
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Wrapper, merge



from keras.utils import CustomObjectScope
from keras.utils.generic_utils import get_custom_objects
from aml_planners.push_planner.forward_models.concrete_dropout import ConcreteDropout


from functools import partial

import keras.losses

keras.losses.custom_loss = ConcreteDropout.heteroscedastic_loss

class DropoutNN(object):

    def __init__(self, config, data=None):
        self._config =  config
        self._Ns = self._config['Ns']
        self._nb_epochs = self._config['nb_epochs']
        self._nb_val_size = self._config['nb_val_size']
        self._nb_features = self._config['nb_features']
        self._Q = self._config['Q']
        self._D = self._config['D']
        self._K_test = self._config['K_test']
        self._nb_reps = self._config['nb_reps']
        self._batch_size = self._config['batch_size']
        self._l = self._config['l']
        self._data = data
        
        get_custom_objects().update({'ConcreteDropout': ConcreteDropout, 'heteroscedastic_loss': ConcreteDropout.heteroscedastic_loss})

        if self._config['load_saved_model']:
            self._model = load_model(self._config['model_path'])
        else:
            self._model = None
        self._ELBO = None

    def gen_data(self, N):
        tmp = np.array(random.sample(self._data, N))
        X = tmp[:,0:self._Q]
        Y = tmp[:,self._Q:]
        return X, Y


    def fit_model(self, nb_epoch, X, Y):
        if K.backend() == 'tensorflow':
            K.clear_session()
        N = X.shape[0]
        wd = self._l**2. / N
        dd = 2. / N
        inp = Input(shape=(self._Q,))
        x = inp
        x = ConcreteDropout(Dense(self._nb_features, activation='relu'), weight_regularizer=wd, dropout_regularizer=dd)(x)
        x = ConcreteDropout(Dense(self._nb_features, activation='relu'), weight_regularizer=wd, dropout_regularizer=dd)(x)
        x = ConcreteDropout(Dense(self._nb_features, activation='relu'), weight_regularizer=wd, dropout_regularizer=dd)(x)
        x = ConcreteDropout(Dense(self._nb_features, activation='relu'), weight_regularizer=wd, dropout_regularizer=dd)(x)
        x = ConcreteDropout(Dense(self._nb_features, activation='relu'), weight_regularizer=wd, dropout_regularizer=dd)(x)
        x = ConcreteDropout(Dense(self._nb_features, activation='relu'), weight_regularizer=wd, dropout_regularizer=dd)(x)
        x = ConcreteDropout(Dense(self._nb_features, activation='relu'), weight_regularizer=wd, dropout_regularizer=dd)(x)
        x = ConcreteDropout(Dense(self._nb_features, activation='relu'), weight_regularizer=wd, dropout_regularizer=dd)(x)
        mean = ConcreteDropout(Dense(self._D), weight_regularizer=wd, dropout_regularizer=dd)(x)
        log_var = ConcreteDropout(Dense(self._D), weight_regularizer=wd, dropout_regularizer=dd)(x)
        out = merge([mean, log_var], mode='concat')
        model = Model(inp, out)

        if self._config['optimizer'] == 'adam':
            optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        elif self._config['optimizer'] == 'sgd':
            optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        
        model.compile(optimizer=optimizer, loss='heteroscedastic_loss')
        assert len(model.layers[1].trainable_weights) == 3  # kernel, bias, and dropout prob
        assert len(model.losses) == 10  # a loss for each Concrete Dropout layer
        hist = model.fit(X, Y, nb_epoch=nb_epoch, batch_size=self._batch_size, verbose=1)
        loss = hist.history['loss'][-1]
        return model, -0.5 * loss, hist.history['loss']  # return ELBO up to const.


    ###eval function
    def logsumexp(self, a):
        a_max = a.max(axis=0)
        return np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max

    def test(self, Y_true, MC_samples):
        """
        Estimate predictive log likelihood:
        log p(y|x, D) = log int p(y|x, w) p(w|D) dw
                     ~= log int p(y|x, w) q(w) dw
                     ~= log 1/K sum p(y|x, w_k) with w_k sim q(w)
                      = LogSumExp log p(y|x, w_k) - log K
        :Y_true: a 2D array of size N x dim
        :MC_samples: a 3D array of size samples K x N x 2*D
        """
        assert len(MC_samples.shape) == 3
        assert len(Y_true.shape) == 2
        k = MC_samples.shape[0]
        N = Y_true.shape[0]
        mean = MC_samples[:, :, :self._D]  # K x N x D
        logvar = MC_samples[:, :, self._D:]
        test_ll = -0.5 * np.exp(-logvar) * (mean - Y_true[None])**2. - 0.5 * logvar - 0.5 * np.log(2 * np.pi)
        test_ll = np.sum(np.sum(test_ll, -1), -1)
        test_ll = self.logsumexp(test_ll) - np.log(k)
        pppp = test_ll / N  # per point predictive probability
        rmse = np.mean((np.mean(mean, 0) - Y_true)**2.)**0.5
        return pppp, rmse

    ##plot function
    def plot(self, X_train, Y_train, X_val, Y_val, means, loss):
        pylab.figure()
        indx = np.argsort(X_val[:, 0])
        _, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = pylab.subplots(1, 8,figsize=(19, 1.5), sharex=False, sharey=False)
        ax1.scatter(X_train[:, 0], X_train[:, 1], c='r')
        ax1.scatter(Y_train[:, 0], Y_train[:, 1], c='g')
        ax1.set_title('Train set')
    #     ax2.plot(X_val[indx, 0], np.mean(means, 0)[indx, 0], color='skyblue', lw=3)
        ax2.scatter(X_val[:, 0], X_val[:, 1], c='r', alpha=0.1,)
        ax2.scatter(Y_val[:, 0], Y_val[:, 1], c='g', alpha=0.2,)
        means_tmp = np.mean(means, 0)
        ax2.scatter(means_tmp[:, 0], means_tmp[:, 1], c='m')
        ax2.set_title('+Predictive mean')
        for mean in means:
            ax3.scatter(mean[:, 0], mean[:, 1], c='gray', alpha=0.2, lw=0)
        ax3.scatter(np.mean(means, 0)[indx, 0], np.mean(means, 0)[indx, 1], color='skyblue', lw=3)
        ax3.set_title('+MC samples on validation X')
        ax4.scatter(X_val[:, 0], X_val[:, 1], c='r', alpha=0.2, lw=0)
        ax4.scatter(Y_val[:, 0], Y_val[:, 1], c='g', alpha=0.2, lw=0)
        ax4.set_title('Validation set')
        
        ax5.plot(loss)
        ax5.set_title('loss plot')
        
        ax6.plot(Y_val[:, 0], c='r')
        ax6.plot(means_tmp[:, 0], c='g')
        ax6.set_title('X')
        
        ax7.plot(Y_val[:, 1], c='r')
        ax7.plot(means_tmp[:, 1], c='g')
        ax7.set_title('Y')
        
        # ax8.plot(Y_val[:, 2], c='r')
        # ax8.plot(means_tmp[:, 2], c='g')
        # ax8.set_title('th')
        pylab.show()


    def run(self):
        results = []
        # get results for multiple N
        for N, nb_epoch in zip(self._Ns, self._nb_epochs):
            # repeat exp multiple times
            rep_results = []
            for i in range(self._nb_reps):
                X, Y = self.gen_data(N + self._nb_val_size)
                X_train, Y_train = X[:N], Y[:N]
                X_val, Y_val = X[N:], Y[N:]
                model, ELBO, loss_history = self.fit_model(nb_epoch, X_train, Y_train)
                MC_samples = np.array([model.predict(X_val) for _ in range(self._K_test)])
                pppp, rmse = self.test(Y_val, MC_samples)  # per point predictive probability
                means = MC_samples[:, :, :self._D]  # K x N
                epistemic_uncertainty = np.var(means, 0).mean(0)
                logvar = np.mean(MC_samples[:, :, self._D:], 0)
                aleatoric_uncertainty = np.exp(logvar).mean(0)
                ps = np.array([K.eval(layer.p) for layer in model.layers if hasattr(layer, 'p')])
                self.plot(X_train, Y_train, X_val, Y_val, means, loss_history)
                rep_results += [(rmse, ps, aleatoric_uncertainty, epistemic_uncertainty)]
            test_mean = np.mean([r[0] for r in rep_results])
            test_std_err = np.std([r[0] for r in rep_results]) / np.sqrt(self._nb_reps)
            ps = np.mean([r[1] for r in rep_results], 0)
            aleatoric_uncertainty = np.mean([r[2] for r in rep_results])
            epistemic_uncertainty = np.mean([r[3] for r in rep_results])
            print "N \t",N 
            print "Epochs \t",nb_epoch
            print "Test mean \t", test_mean
            print "Test standard error \t", test_std_err 
            print "Dropout p \t",ps
            print "Aleatoric uncertainty \t",aleatoric_uncertainty**0.5
            print "Epistemic uncertainty \t",epistemic_uncertainty**0.5
            sys.stdout.flush()
            results += [rep_results]
        return results

    def fit(self, X, Y, N=390):

        if self._config['load_saved_model']:
            print "Loading saved model from ", self._config['model_path']
            self._model = load_model(self._config['model_path'], custom_objects={'custom_loss': ConcreteDropout.heteroscedastic_loss})
        else:
            if self._data is not None:
                X, Y = self.gen_data(N + self._nb_val_size)
            X_train, Y_train = X[:N], Y[:N]
            self._model, self._ELBO, loss_history = self.fit_model(self._nb_epochs[0], X_train, Y_train)

            if self._config['save_model']:
                self._model.save(self._config['model_path'])

    def predict(self, X_val, Y_val=None, seperate_uncertainity=False):

        MC_samples = np.array([self._model.predict(X_val) for _ in range(self._K_test)])

        if Y_val is not None:
            pppp, rmse = self.test(Y_val, MC_samples)  # per point predictive probability
            return pppp, rmse

        means = MC_samples[:, :, :self._D]  # K x N
        logvar = np.mean(MC_samples[:, :, self._D:], 0)

        prediction = np.mean(means, 0)
        epistemic_uncertainty = np.array([np.var(means[:,k,:]) for k in range(prediction.shape[0])]) 
        aleatoric_uncertainty = np.array([np.sum(np.exp(logvar[k,:]))  for k in range(prediction.shape[0])])

        if seperate_uncertainity:
            return prediction, epistemic_uncertainty, aleatoric_uncertainty
        else:
            return prediction, epistemic_uncertainty+aleatoric_uncertainty



