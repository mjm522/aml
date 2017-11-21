import GPy
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C


np.random.seed(1)

class GPModel(object):

    def __init__(self, out_dim = 1):

        # self._kernel = RBF(2.5, (1e-5, 1e5)) # working one
        # self._kernel = C(10.0, (1e-4, 1e2)) * RBF(1.0, (1e-10, 1e7)) #+ WhiteKernel(noise_level=0.000000001) #
        #C(1e-20, (1e-25, 1e-15)) * 
        
        # workshop
        # self._kernel = C(1e-3, (1e-5, 1e1))* RBF(length_scale=2.0)

        self._kernel = RBF(.01, (1e-5, 1e-1))
        # ConstantKernel(1e-20, (1e-25, 1e-15))* RBF(length_scale=1)
        # Instanciate a Gaussian Process model
        #alpha=(dy / y) ** 2

        self._out_dim = out_dim
        # self._gps = [ GaussianProcessRegressor(kernel=self._kernel, optimizer=None,
        #                               n_restarts_optimizer=10) for i in range(out_dim)]


    def fit(self, X, y):

        ## Fitting each gp individually
        # for i in range(len(self._gps)):
        #     self._gps[i].fit(X,y[:,i])

        # for i in range(y.shape[1]):
        #     self
        self._m_full_x  = GPy.models.GPRegression(X=X,  Y=y[:,0][:,None], noise_var=.1)
        self._m_full_y  = GPy.models.GPRegression(X=X,  Y=y[:,1][:,None], noise_var=.1)
        self._m_full_th = GPy.models.GPRegression(X=X,  Y=y[:,2][:,None], noise_var=.1)

        print "Log likelihood BEFORE optimization"
        print "Log likelihood of m_full_x \t", self._m_full_x.log_likelihood()
        print "Log likelihood of m_full_y \t", self._m_full_y.log_likelihood()
        print "Log likelihood of m_full_th \t", self._m_full_th.log_likelihood()

        self._m_full_x.optimize('bfgs')
        self._m_full_y.optimize('bfgs')
        self._m_full_th.optimize('bfgs')

        print "Log likelihood AFTER optimization"
        print "Log likelihood of m_full_x \t", self._m_full_x.log_likelihood()
        print "Log likelihood of m_full_y \t", self._m_full_y.log_likelihood()
        print "Log likelihood of m_full_th \t", self._m_full_th.log_likelihood()


    def predict(self, x):

        # mus = [None]*self._out_dim
        # sigmas = [None]*self._out_dim

        # for i in range(len(self._gps)):
        #     mus[i], sigmas[i] = self._gps[i].predict(x, return_std=True)

        mus_x,    sigmas_x  = self._m_full_x.predict(x)
        mus_y,    sigmas_y  = self._m_full_y.predict(x)
        mus_th,   sigmas_th = self._m_full_th.predict(x)

        return np.hstack([mus_x, mus_y, mus_th]), np.hstack([sigmas_x, sigmas_y, sigmas_th])


