import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from aml_io.io_tools import save_data, load_data
from aml_opt.pi_traj_opt.pi_traj_opt import PITrajOpt
from aml_playground.peg_in_hole.costs.cost_var_stiff import VarStiffnessCost

np.random.seed(42)

class VaribaleStiffnessFinder(object):

    def __init__(self, config):

        config['start'] = config['start_goal']['start']
        config['goal']  = config['start_goal']['goal']
        self._KPx_min, self._KPy_min, self._KPz_min = config['state_constraints']['min']
        self._KPx_max, self._KPy_max, self._KPz_max = config['state_constraints']['min']

        cost_fn = VarStiffnessCost()

        self._opt = PITrajOpt(config, cost_fn, self.visualize)
        self._init_traj = self._opt._init_traj

        self._fig = plt.figure("Var Stiff")
        self._ax = self._fig.add_subplot(111, projection='3d')
        
        plt.ion()

  
    def visualize(self, traj, traj_samples=None):

        if traj_samples is not None:
            for i in range(traj_samples.shape[0]):
                for k in range(traj_samples.shape[2]):
                    traj_samples[i,:,k] = self._opt.savitsky_gollay_filter(traj_samples[i,:,k])

        plt.clf()
        self._ax = self._fig.add_subplot(111, projection='3d')
        
        if (traj_samples is not None):
            for i in range(traj_samples.shape[0]):
                self._ax.plot(traj_samples[:,:,0].flatten(), 
                              traj_samples[:,:,1].flatten(), 
                              traj_samples[:,:,2].flatten(), alpha=0.2)

        self._ax.plot(traj[:,0], traj[:,1], traj[:,2], color='k', linewidth=8.)

        plt.xlim(self._KPx_min, self._KPy_min, self._KPz_min)
        plt.ylim(self._KPx_max, self._KPy_max, self._KPz_max)
        plt.draw()
        plt.pause(0.00001)

    def run(self):
        self._opt.run()



