import numpy as np

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