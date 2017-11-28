import os
import numpy as np
import matplotlib.pyplot as plt
from config import ex_config as config
from aml_io.io_tools import save_data, load_data
from aml_opt.pi_traj_opt.pi_traj_opt import PITrajOpt

class TrajCost():
    def __init__(self, heatmap_data_file):
        self._heatmap_data_file = heatmap_data_file

    def __call__(self, traj):
        return self.get(traj)

    def get(self, traj):

        x_traj = traj[:,0]
        y_traj = traj[:,1]

        cost = np.zeros(traj.shape[0])
        i = 0
        for x, y in zip(x_traj, y_traj):
            if (x/7.) >= 1 or (y/7.) >= 1:
                value = 1e3
            else:
                value = self._heatmap_data_file[int(100.*x/7.),int(100*y/7.)]

            cost [i]= value
            i += 1 
        cost = cost - np.min(cost)

        cost = 1./ (1. + np.exp(-cost))

        cost = cost - np.min(cost)

        return cost

class LowCostPathFinder(object):

    def __init__(self, config, heatmap_data_file):

        config['start'] = config['start_goal']['start']
        config['goal']  = config['start_goal']['goal']
        self._x_min = config['state_constraints']['min'][0]
        self._x_max = config['state_constraints']['max'][0]
        self._y_min = config['state_constraints']['min'][1]
        self._y_max = config['state_constraints']['max'][1]

        self._heatmap_data_file = load_data(heatmap_data_file)

        cost_fn = TrajCost(self._heatmap_data_file)

        self._opt = PITrajOpt(config, cost_fn, self.visualize)
        self._init_traj = self._opt._init_traj

        plt.imshow(self._heatmap_data_file, origin='lower', interpolation='none', extent=[self._x_min,self._x_max,
                                                                                         self._y_min,self._y_max])

    
        plt.colorbar()
        plt.plot(self._init_traj[0,:], self._init_traj[1,:], color='g', linewidth=3)
        plt.ion()

    def visualize(self, traj):
        x_traj = traj[:,0]
        y_traj = traj[:,1]
        plt.clf()
        plt.imshow(self._heatmap_data_file, origin='lower', interpolation='none', extent=[self._x_min,self._x_max,
                                                                                          self._y_min,self._y_max])
        plt.colorbar()
        plt.plot(x_traj, y_traj, color='m', linewidth=3.)
        plt.draw()
        plt.pause(0.00001)

    def run(self):
        self._opt.run()


def main():

    heatmap_data_file = os.environ['AML_DATA'] + '/aml_planners/push_planner/single_push_planner/heat_maps/baxter_heatmap.pkl'

    lcpf = LowCostPathFinder(config=config, heatmap_data_file=heatmap_data_file)
    lcpf.run()

if __name__ == '__main__':
    main()

