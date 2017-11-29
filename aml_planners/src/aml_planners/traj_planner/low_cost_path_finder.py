import os
import numpy as np
from cost import TrajCost
import matplotlib.pyplot as plt
from config import ex_config as config
from aml_io.io_tools import save_data, load_data
from aml_opt.pi_traj_opt.pi_traj_opt import PITrajOpt

np.random.seed(42)

class LowCostPathFinder(object):

    def __init__(self, config, heatmap_data_file):

        config['start'] = config['start_goal']['start']
        config['goal']  = config['start_goal']['goal']
        self._x_min, self._y_min = config['state_constraints']['min']
        self._x_max, self._y_max = config['state_constraints']['max']

        self._heatmap_data_file = load_data(heatmap_data_file)

        cost_fn = TrajCost(self._heatmap_data_file)

        self._opt = PITrajOpt(config, cost_fn, self.visualize)
        self._init_traj = self._opt._init_traj

        plt.imshow(self._heatmap_data_file, origin='lower', interpolation='none', extent=[self._x_min,self._x_max,
                                                                                         self._y_min,self._y_max])

    
        plt.colorbar()
        plt.plot(self._init_traj[0,:], self._init_traj[1,:], color='g', linewidth=3)
        plt.ion()
  
    def visualize(self, traj, traj_samples=None):
        x_traj = traj[:,0]
        y_traj = traj[:,1]
        plt.clf()
        plt.imshow(self._heatmap_data_file, origin='lower', interpolation='none', extent=[self._x_min,self._x_max,
                                                                                         self._y_min,self._y_max])
        plt.colorbar()
        
        if (traj_samples is not None):
            for i in range(traj_samples.shape[0]):
                plt.plot(self._opt.savitsky_gollay_filter(traj_samples[i,:,0].flatten()), 
                         self._opt.savitsky_gollay_filter(traj_samples[i,:,1].flatten()), alpha=0.2)

        plt.plot(x_traj.flatten(), y_traj.flatten(), color='k', linewidth=8.)

        plt.xlim(self._x_min, self._x_max)
        plt.ylim(self._y_min, self._y_max)
        plt.draw()
        plt.pause(0.00001)

    def run(self):
        self._opt.run()


def main():

    heatmap_data_file = os.environ['AML_DATA'] + '/aml_planners/traj_planner/heat_maps/heatmap_good.pkl'

    lcpf = LowCostPathFinder(config=config, heatmap_data_file=heatmap_data_file)
    lcpf.run()

if __name__ == '__main__':
    main()

