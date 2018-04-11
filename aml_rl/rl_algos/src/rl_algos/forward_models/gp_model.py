import GPy
import numpy as np

class GPModel():

    def __init__(self, x_dim, kernel_var=0.1, kernel_len=0.1):

        self._x_dim = x_dim
        self._kernel = GPy.kern.RBF(input_dim=self._x_dim, variance=kernel_var, lengthscale=kernel_len)

    def fit(self, X, Y):
        
        self._model  = GPy.models.GPRegression(X=X,  Y=Y)#, kernel=self._kernel)
        self._model.optimize('bfgs')

    def predict(self, x):

        if x.ndim == 1:
            x = x[None,:]

        mu, sigma = self._model.predict(x)

        return mu, sigma