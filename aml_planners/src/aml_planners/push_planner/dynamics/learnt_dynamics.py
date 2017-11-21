import sys, os
import numpy as np
import pandas as pd

from aml_planners.push_planner.utilities.utils import sigmoid
from aml_planners.push_planner.dynamics.dynamics import Dynamics
from aml_planners.push_planner.utilities.utils_ros import add_new_dim
from aml_planners.push_planner.dynamics.box2d_dynamics import Box2DDynamics
from aml_planners.push_planner.exp_params.experiment_params import experiment_config

'''
data_input: xi,yi,thetai,dxi,dyi,dthetai,a_px,a_py,a_f,a_alpha

data_output: xf,yf,thetaf,dxf,dyf,dthetaf
'''

class LearnedDynamics(Dynamics):

    def __init__(self, config, pre_process=False, add_extra_dim=False):

        Dynamics.__init__(self, config['dt'])


        self._config = config

        ## Get the data
        self._df = pd.DataFrame.from_csv(config['data_path'])

        self._add_extra_dim = add_extra_dim
        
        self._X = np.array(self._df[['xi','yi', 'thetai']])
        
        actions = np.array(self._df[['action']])

        if self._add_extra_dim:
            actions = add_new_dim(actions)

        if actions.ndim == 1:
            actions =  actions[:,None]

        y = np.array(self._df[['xf','yf', 'thetaf']])

        self._y = y - self._X

        self._X =  np.hstack([self._X, actions])


        ##for converting the action
        self._box2d_world = Box2DDynamics(config)


        # Instantiate forward model
        self._model = config['forward_model'](config)

        self._loss = self._model.fit(self._X, self._y)

    def dynamics(self, x, u):

        # u = np.multiply(self._config['cmd_filter'],u)
        # action = [0., 0., u[0], u[1]] #px, py, f_x, f_y 
        action = sigmoid(u)

        if self._add_extra_dim:
            action = add_new_dim(action)
  
        
        XS_Test = (np.r_[x[:3], action.ravel()])[None, :] #to avoid deprecated warning of scikit learn
        mus, sigmas = self._model.predict(XS_Test)

        #this is to match the box2d dynamics and to sent to the box2d world
        converted_action = self._box2d_world.convert_push_action(action[0])

        # mus = mus[0]

        # y = np.r_[mus[2]*mus[:2], 0.,0.,0,0.]

        # mus = y

        mus = np.r_[x[:3]+mus.flatten(), 0., 0., 0.]

        return mus, sigmas.squeeze(), converted_action


##########################TEST CODE#################################

def main():

    bxDyn = LearnedDynamics(experiment_config)
    state = [5.,5.,0.,0.,0.,0.]
    u = [0.,0.,2.0, 2.0]

    for k in range(100):
        state, sigmas = bxDyn.dynamics(state,u)
        state = state.squeeze()
        sigmas = sigmas.squeeze()
        print "State \t:", np.round(state,2), "Sigma \t:", sigmas

if __name__ == '__main__':
    main()