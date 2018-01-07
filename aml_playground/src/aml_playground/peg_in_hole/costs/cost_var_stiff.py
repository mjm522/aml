import numpy as np

class VarStiffnessCost():
    def __init__(self):
        pass

    def __call__(self, traj):
        return self.get(traj)

    def get(self, traj):

        KPx_traj = traj[:,0]
        KPy_traj = traj[:,1]
        KPz_traj = traj[:,2]

        cost = np.zeros(traj.shape[0])
        i = 0
        for KPx, KPy, KPz in zip(KPx_traj, KPy_traj, KPz_traj):
            
            value = KPx + KPy + KPz

            cost [i]= value
            i += 1 
        cost = cost - np.min(cost)

        cost = 1./ (1. + np.exp(-cost))

        cost = cost - np.min(cost)

        return cost