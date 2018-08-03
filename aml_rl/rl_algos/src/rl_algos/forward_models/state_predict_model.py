import numpy as np
from collections import deque
from rl_algos.forward_models.gp_model import GPModel
from rl_algos.forward_models.reps_fwd_model import REPSFwdModel

class StatePredictModel(REPSFwdModel):

    def __init__(self, x_dim, y_dim, num_data_points=30):

        self._x_dim = x_dim
        self._y_dim = y_dim
        self._x_data  = deque(maxlen=num_data_points)
        self._y_data = deque(maxlen=num_data_points)

        self.state_model = [GPModel(x_dim=x_dim) for _ in range(self._y_dim)]


    def add_data(self, x, y):

        assert len(x) == self._x_dim
        assert len(y) == self._y_dim
        
        self._x_data.append(x)
        self._y_data.append(y)

    def fit(self):

        X = np.asarray(self._x_data)
        Y = np.asarray(self._y_data)

        for i, model in enumerate(self.state_model):                      
    
            model.fit(X=X,  Y=Y[:, i][:, None])


    def predict(self, x):

        mu_y = np.zeros(self._y_dim)
        sigma_y =  np.zeros(self._y_dim)


        return 3.*x

        for i, model in enumerate(self.state_model): 

            mu_y[i], sigma_y[i] = model.predict(x)

        return mu_y
