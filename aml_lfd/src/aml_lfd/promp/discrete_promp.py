import numpy as np
import numpy.matlib as npm
from scipy.interpolate import interp1d


class DiscretePROMP(object):

    def __init__(self, data, num_bfs=30, bfs_sigma=0.05):

        """
        Constructor of the class
        Args: 
        data: a list of demonstrated trajectories
        num_bfs = number of basis functions
        bfs_sigma = width of the basis function
        """

        #list to store all demo trajectories
        self._demo_trajs = data
        
        #number of demos available
        self._num_demos = len(self._demo_trajs)
        #lenght of each demonstrations
        
        self._traj_len  = len(self._demo_trajs[0])
        #time step
        self._dt  = 1./self._traj_len
        
        #list to store all demo traj velocities
        self._Ddemo_trajs = self.compute_velocities()

        #number of basis function
        self._n_bfs = num_bfs
        #variance of the basis function
        self._bfs_sigma = bfs_sigma
        #centers of the basis function
        self._bfs_centres = np.linspace(0, 1, self._n_bfs)

        #list that stores all the weights
        self._W = []

        #mean and sigma of the weights
        self._mean_W  = None
        self._sigma_W = None

        #compute the basis functions
        self._Phi, self._PhiD, self._PhiDD, self._phase = self.generate_basis_function(tau=1.)

        #via points
        self._viapoints = []


    def generate_basis_function(self, tau):

        phase = self.compute_phase(tau=tau)

        # basis functions
        phase_minus_centre = np.array(map(lambda x: x - self._bfs_centres, np.tile(phase._z, (self._n_bfs, 1)).T)).T

        #basis function
        Phi   = np.exp(-0.5 *np.power(phase_minus_centre/self._bfs_sigma, 2)) / (np.sqrt(2.*np.pi)*self._bfs_sigma)
        
        #first derivative of basis function
        PhiD  = np.multiply(Phi,  -phase_minus_centre/(self._bfs_sigma ** 2))

        #second derivative of basis function
        PhiDD = Phi/(-self._bfs_sigma ** 2) + np.multiply(-phase_minus_centre/(self._bfs_sigma ** 2), PhiD)

        #for normalization purposes
        sum_bfs    = np.sum(Phi,   axis=0)
        sum_bfsD   = np.sum(PhiD,  axis=0)
        sum_bfsDD  = np.sum(PhiDD, axis=0)

        # normalize
        PhiD_normalized = ( (PhiD * sum_bfs - Phi * sum_bfsD) * 1./np.power(sum_bfs, 2) ) 
        
        Phi_normalized  = Phi/sum_bfs[None, :]
 
        tmp1 = Phi * (2 * np.power(sum_bfsD, 2) - np.multiply(sum_bfs, sum_bfsDD))

        tmp2 = tmp1 + PhiDD * np.power(sum_bfs, 2) - 2 * PhiD * sum_bfs * sum_bfsD

        #normalize acceleration
        PhiDD_normalized = tmp2 * (1./(np.power(sum_bfs, 3)))

        #adding phase dependency
        PhiDD_normalized = PhiDD_normalized * np.power(phase._Dz, 2)  + PhiD_normalized * phase._DDz
        PhiD_normalized  = PhiD_normalized * phase._Dz

        return Phi_normalized, PhiD_normalized, PhiDD_normalized, phase


    def compute_velocities(self):
        """
        function to add new demo to the list
        param: traj : a uni dimentional numpy array
        """
        Ddemo_trajs = []

        for demo_traj in self._demo_trajs:
            d_traj = np.diff(demo_traj, axis=0)/self._dt
            #append last element to adjust the length
            d_traj = np.hstack([d_traj, d_traj[-1]])
            #add it to the list
            Ddemo_trajs.append(d_traj)


    def compute_phase(self, tau):
        """
        This function is for adding the temporal scalability for 
        the basis function
        """

        class Phase():

            def __init__(self, dt, tau):

                #phase variable
                z   = np.arange(dt, (1.+dt)*tau, dt)
                #derivative of phase variable
                Dz  = np.diff(z)/dt
                Dz  = np.hstack([Dz, Dz[-1]])
                #second derivative of phase variable
                DDz = np.diff(Dz)/dt
                DDz = np.hstack([DDz, DDz[-1]])

                self._z   = z
                self._Dz  = Dz
                self._DDz = DDz

            def __call__(self):
                return self

        return Phase(self._dt, tau=tau)


    def train(self):
        """
        This function finds the weights of the 
        given demonstration by first interpolating them
        to 0-1 range and then finding the kernel weights
        corresponding to each trajectory
        """

        for demo_traj in self._demo_trajs:
            #linear interpolation of the demonstrated trajectory
            interpolate = interp1d(np.linspace(0, 1, len(demo_traj)), demo_traj, kind='cubic')
            #strech the trajectory to fit 0 to 1
            stretched_demo = interpolate(self._phase._z)[None,:]

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
        return self._Phi[:, t_index], self._PhiD[:, t_index]


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


    def generate_trajectory(self, tau=1., randomness=1e-4):
        """
        Outputs a trajectory
        :param randomness: float between 0. (output will be the mean of gaussians) and 1. (fully randomized inside the variance)
        :return: a 1-D vector of the generated points
        """
        new_mean_W   = self._mean_W
        new_sigma_W  = self._sigma_W

        for viapoint in self._viapoints:
            # basis functions at observed time points
            time_stamps = np.tile(viapoint['t']*tau, (self._n_bfs, 1)).T

            PhiT = np.exp(-.5 * (np.array(map(lambda x: x - self._bfs_centres, time_stamps)).T ** 2 / (self._bfs_sigma ** 2)))
            PhiT = PhiT / sum(PhiT)

            # Conditioning
            aux = viapoint['traj_point_sigma'] + np.dot(np.dot(PhiT.T, new_sigma_W), PhiT)
            new_mean_W = new_mean_W + np.dot(np.dot(new_sigma_W, PhiT) * 1 / aux, (viapoint['traj_point'] - np.dot(PhiT.T, new_mean_W)))  # new weight mean conditioned on observations
            new_sigma_W = new_sigma_W - np.dot(np.dot(new_sigma_W, PhiT) * 1 / aux, np.dot(PhiT.T, new_sigma_W))

        sample_W = np.random.multivariate_normal(new_mean_W, randomness*new_sigma_W, 1).T

        return np.dot(self._Phi.T, sample_W)


    def temporal_scaling(self, tau=1., randomness=1e-4):

            print PhiT.shape


            PhiT, PhiTD, PhiTDD, _ = self.generate_basis_function(tau=tau)

            print PhiT.shape

