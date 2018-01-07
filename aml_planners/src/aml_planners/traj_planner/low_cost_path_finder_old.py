import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from aml_io.io_tools import save_data, load_data

np.random.seed(42)

class LowCostPathFinder(object):

    def __init__(self, sg, heatmap_data_file, state_constraints, no_rollouts=100, no_timesteps=100):
        self._sg = sg
        self._heatmap_data_file = load_data(heatmap_data_file)
        self._N = no_timesteps
        self._K = no_rollouts
        self._h = 10
        self._gain = 1e-1
        self._max_iter = 61

        self._traj_file_name = None

        self._x_min = state_constraints['x']['min']
        self._y_min = state_constraints['y']['min']

        self._x_max = state_constraints['x']['max']
        self._y_max = state_constraints['y']['max']

        self._init_path_x = np.linspace(sg['start'][0], sg['goal'][0], self._N)
        self._init_path_y = np.linspace(sg['start'][1], sg['goal'][1], self._N)

        plt.imshow(self._heatmap_data_file, origin='lower', interpolation='none', extent=[self._x_min,self._x_max,
                                                                                         self._y_min,self._y_max])
        plt.colorbar()
        plt.plot(self._init_path_x, self._init_path_y, color='g', linewidth=3)
        plt.ion()


    def get_cost_traj(self, x_traj, y_traj):
        cost = np.zeros(len(x_traj))
        i = 0
        for x,y in zip(x_traj, y_traj):
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


    def put_x_state_constraints(self, traj):
        traj[traj<self._x_min] = self._x_min
        traj[traj>self._x_max] = self._x_max

        return traj[:, None]

    def put_y_state_constraints(self, traj):
        traj[traj<self._y_min] = self._y_min
        traj[traj>self._y_max] = self._y_max

        return traj[:, None] 

    def get_traj_samples(self, x_traj, y_traj, gain=1e-0):

        x_traj_samples =  np.zeros([self._K, self._N, 1])
        y_traj_samples =  np.zeros([self._K, self._N, 1])
        cost_traj_samples = np.zeros([self._K, self._N])
        dx_traj_samples = np.random.randn(self._K, self._N, 1)
        dy_traj_samples = np.random.randn(self._K, self._N, 1)

        for k in range(self._K):
            
            x_traj_samples[k,:,:] = self.put_x_state_constraints( (x_traj + gain*dx_traj_samples[k,:,:]).flatten() )
            y_traj_samples[k,:,:] = self.put_y_state_constraints( (y_traj + gain*dy_traj_samples[k,:,:]).flatten()  )

            cost_traj_samples[k,:] = self.get_cost_traj(x_traj_samples[k,:,:], y_traj_samples[k,:,:])

        return x_traj_samples, y_traj_samples, dx_traj_samples, dy_traj_samples, cost_traj_samples

    def compute_traj_change(self, cost_samples, del_traj_samples):
        exp_cost_samples = np.exp(-self._h*cost_samples)
        denominators = np.sum(exp_cost_samples, axis=0)
        del_traj_samples = del_traj_samples/denominators[None,:,None]
        del_traj_samples = np.multiply(exp_cost_samples[:, :, None], del_traj_samples)

        return np.sum(del_traj_samples, axis=0)

    def savitsky_gollay_filter(self, traj):
        return savgol_filter(x=traj, window_length=5, polyorder=2)

    def modify_traj(self, x_traj, y_traj):
        x_traj_samples, y_traj_samples, \
        dx_traj_samples, dy_traj_samples, cost_traj_samples = self.get_traj_samples(x_traj, y_traj)

        tmp_x =  self.put_x_state_constraints(  (x_traj + 1e-1*self.compute_traj_change(cost_traj_samples, dx_traj_samples)).flatten() )[1:self._N-1, :] 
        tmp_y =  self.put_x_state_constraints(  (y_traj + 1e-1*self.compute_traj_change(cost_traj_samples, dy_traj_samples)).flatten() )[1:self._N-1, :] 

        x_traj[1:self._N-1, :] = self.savitsky_gollay_filter(tmp_x.flatten())[:,None]
        y_traj[1:self._N-1, :] = self.savitsky_gollay_filter(tmp_y.flatten())[:,None]

        cost_traj = self.get_cost_traj(x_traj, y_traj)

        self.visualize(x_traj, y_traj)
        
        return x_traj, y_traj, np.sum(cost_traj)

    def run(self):

        xf_traj =  self._init_path_x.copy()[:,None]
        yf_traj =  self._init_path_y.copy()[:,None]
        
        for t in range(self._max_iter):

            xf_traj, yf_traj, cost = self.modify_traj(xf_traj, yf_traj)

            print "Iteration %d and cost is %f"%(t, cost)
        
        if self._traj_file_name is not None:
            save_data(np.hstack([xf_traj, yf_traj]), self._traj_file_name)
        
        raw_input("Waiting...")

    def visualize(self, x_traj, y_traj):
        plt.clf()
        plt.imshow(self._heatmap_data_file, origin='lower', interpolation='none', extent=[self._x_min,self._x_max,
                                                                                         self._y_min,self._y_max])
        plt.colorbar()
        plt.plot(x_traj.flatten(), y_traj.flatten(), color='m', linewidth=3.)
        plt.draw()
        plt.pause(0.00001)


def main():

    sg ={
    'start':np.array([0.75, 6.75]),
    'goal':np.array([6.5, 0.75]),
    'obstacle':np.array([3.5,3.5]),
    'r_obs':1.0,
    }

    state_constraints={
    'x':{'min':0., 'max':7.},
    'y':{'min':0., 'max':7.},
    }


    heatmap_data_file = os.environ['AML_DATA'] + '/aml_planners/traj_planner/heat_maps/heatmap_good.pkl'

    lcpf = LowCostPathFinder(sg=sg, 
                             state_constraints=state_constraints, 
                             heatmap_data_file=heatmap_data_file, 
                             no_rollouts=100, 
                             no_timesteps=100)
    lcpf.run()

if __name__ == '__main__':
    main()

