import numpy as np
import numpy.matlib as npm
from scipy.interpolate import interp1d


class DiscretePROMP(object):

    def __init__(self,):
        #list to store all demo trajectories
        self._demo_trajs = []

        #list to store all demo traj velocities
        self._Ddemo_trajs = []

        #number of basis function
        self._n_bfs = 11

        #variance of the basis function
        self._bfs_sigma = 0.05

        #number of samples per demo
        self._num_samples = 100

        #time step
        self._dt  = 1./self._num_samples

        #list that stores all the weights
        self._W = []

        #for rescaling the trajectories to 0 and 1
        self._x = self.canonical_system() #np.linspace(0, 1, self._num_samples)

        #centers of the basis function
        self._bfs_centres = np.arange(0, self._n_bfs) / (self._n_bfs - 1.0)

        # basis functions
        self._Phi = np.exp(-.5 * (np.array(map(lambda x: x - self._bfs_centres, np.tile(self._x, (self._n_bfs, 1)).T)).T ** 2 / (self._bfs_sigma ** 2)))
        
        # normalize
        self._Phi /= sum(self._Phi)

        #via points
        self._viapoints = []


    def add_demo_traj(self, traj):
        """
        function to add new demo to the list
        param: traj : a uni dimentional numpy array
        """
        self._demo_trajs.append(traj)
        #numerical differentiation to find velocity
        d_traj = np.diff(traj, axis=0)/self._dt
        #append last element to adjust the length
        d_traj = np.hstack([d_traj, d_traj[-1]])
        #add it to the list
        self._Ddemo_trajs.append(d_traj)


    def canonical_system(self, tau=1., alpha=1.):
        # # canonical system

        x = np.linspace(0, 1, self._num_samples)

        return x


    def train(self):

        for demo_traj in self._demo_trajs:
            #linear interpolation of the demonstrated trajectory
            interpolate = interp1d(np.linspace(0, 1, len(demo_traj)), demo_traj, kind='cubic')
            #strech the trajectory to fit 0 to 1
            stretched_demo = interpolate(self._x)[None,:]

            #compute the weights of the trajectory using the basis function
            w_demo_traj = np.dot(np.linalg.inv(np.dot(self._Phi, self._Phi.T)), np.dot(self._Phi, stretched_demo.T)).T  # weights for each trajectory
            
            #append the weights to the list
            self._W.append(w_demo_traj.copy())

        self._W =  np.asarray(self._W).squeeze()
        
        # mean of weights
        self._mean_W = np.mean(self._W, axis=0)
        
        # covariance of weights
        # w1 = np.array(map(lambda x: x - self._mean_W.T, self._W))
        # self._sigma_W = np.dot(w1.T, w1)/self._W.shape[0]

        self._sigma_W = np.cov(self._W.T)


    def clear_viapoints(self):
        """
        delete the already stored via points
        """
        del self._viapoints[:]

    def add_viapoint(self, t, traj_point, traj_point_sigma=1e-6):
        """
        Add a viapoint to the trajectory
        Observations and corresponding basis activations
        :param t: timestamp of viapoint
        :param traj_point: observed value at time t
        :param sigmay: observation variance (constraint strength)
        :return:
        """
        self._viapoints.append({"t": t, "traj_point": traj_point, "traj_point_sigma": traj_point_sigma})

    def set_goal(self, traj_point, traj_point_sigma=1e-6):
        """
        this function is used to set the goal point of the 
        discrete promp. The last value at time step 1
        """
        self.add_viapoint(1., traj_point, traj_point_sigma)

    def set_start(self, traj_point, traj_point_sigma=1e-6):
        """
        this function is used to set the start point of the 
        discrete promp. The last value at time step 0
        """
        self.add_viapoint(0., traj_point, traj_point_sigma)


    def get_mean(self, t_index):
        """
        function to compute mean of a point at a 
        particular time instant
        """
        mean = np.dot(self._Phi.T, self._mean_W)
        return mean[t_index]

    def get_basis(self, t_index):
        """
        returns the basis at a particular instant
        """
        return self._Phi[:, t_index], None


    def get_traj_cov(self):
        """
        return the covariance of a trajectory
        """
        return np.dot(self._Phi.T, np.dot(self._sigma_W, self._Phi))


    def get_std(self):
        """
        standard deviation of a trajectory
        """
        std = 2 * np.sqrt(np.diag(np.dot(self._Phi.T, np.dot(self._sigma_W, self._Phi))))
        return std

    def get_bounds(self, t_index):
        """
        compute bounds of a value at a specific time index
        """
        mean = self.get_mean(t_index)
        std  = self.get_std()
        return mean - std, mean + std


    def generate_trajectory(self, tau=1., randomness=1e-10):
        """
        Outputs a trajectory
        :param randomness: float between 0. (output will be the mean of gaussians) and 1. (fully randomized inside the variance)
        :return: a 1-D vector of the generated points
        """
        new_mean_W   = self._mean_W
        new_sigma_W  = self._sigma_W

        for viapoint in self._viapoints:
            # basis functions at observed time points
            time_stamps = np.tile(viapoint['t'], (self._n_bfs, 1)).T
            time_stamps = 1./tau*time_stamps
            PhiT = np.exp(-.5 * (np.array(map(lambda x: x - self._bfs_centres, time_stamps)).T ** 2 / (self._bfs_sigma ** 2)))
            PhiT = PhiT / sum(PhiT)  

            # Conditioning
            aux = viapoint['traj_point_sigma'] + np.dot(np.dot(PhiT.T, new_sigma_W), PhiT)
            new_mean_W = new_mean_W + np.dot(np.dot(new_sigma_W, PhiT) * 1 / aux, (viapoint['traj_point'] - np.dot(PhiT.T, new_mean_W)))  # new weight mean conditioned on observations
            new_sigma_W = new_sigma_W - np.dot(np.dot(new_sigma_W, PhiT) * 1 / aux, np.dot(PhiT.T, new_sigma_W))

        sample_W = np.random.multivariate_normal(new_mean_W, randomness*new_sigma_W, 1).T
        return np.dot(self._Phi.T, sample_W)
