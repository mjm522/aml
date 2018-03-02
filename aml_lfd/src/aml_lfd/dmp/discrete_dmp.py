import numpy as np
import numpy.matlib as npm
# from aml_lfd.lfd import LfD
from scipy.interpolate import interp1d


class DiscreteDMP(object):
    """
    The DMP class that implements discrete version of the theory
    """

    def __init__(self, config):
        """
        Class constructor
        Args: 
        the config dictionary contains following entities
        dt : time step 
        rbf_num : number of basis functions
        end_time : the last time of the trajectory
        type : what type of forcing function, whether it should allow external disturbance, or the one that allows stalling
                type 1: standard dmp
                type 2: 
                type 3: allows phase stop
        dof : number of degrees of freedom for the robot
        K: proporational gain of DMP
        D: derivative gain of DMP
        tau: temporal scaling parameter of the DMP
        ax: the time constant of the canonical system
        """
        
        # LfD.__init__(self, config)
        self._config = config
        # self._end_time = config['end_time']
        self._dt = config['dt']
        
        dc = 1./(config['rbf_num'] -1 )
        #centers of the basis finction
        centers = np.arange(1., 0., -dc)[None, :]
        # create the basis functions
        self._kernel_fcn = self.create_kernel_fcn(centers, 1)
        #DMPs are time stamped, one goal per time stamp.
        self._time_stamps = np.arange(0, config['end_time'], self._dt)
        #whether to add force of not
        self._type = config['type']
        #number of DOF of the system, we need one DMP per dimension.
        self._dof = config['dof']
        #to store teh demonstrations
        self._traj_data = None
        #the weight of a DMP trained.
        self._Ws = None

    def load_demo_trajectory(self, trajectory):
        """
        this function for storing demo trajectory
        Args: 
        trajectory : arrray of [number of dof, data points]
        Returns:
        none, store the trajectories
        """
        if trajectory.shape[1] != self._dof:
            raise("The trajectory does not have same number of dimensions as specified")
        self._demo_traj = np.vstack([trajectory, npm.repmat(trajectory[-1,:],20,1)])
        self._traj_data = self.process_trajectory()
        self._config['original_scaling'] = self._traj_data[-1,1:] - self._traj_data[1,1:] + 1e-5

    def get_time_stamps(self):
        """
        function to get the time stamps
        Args: None
        Returns : None
        """
        return np.arange(0, self._end_time, self._dt)


    def process_trajectory(self):
        """
        this function processes trajectories. a incoming trajectories are 
        re-scaled to fit [0, 1] using a 1D linear interpolation
        Args: None
        Returns: None 
        """
        
        traj_data = np.zeros([self._time_stamps.shape[0], self._dof+1])
        traj_data[:,0] = self._time_stamps
        
        for ID in range(self._dof):


            traj = self._demo_traj[:len(self._time_stamps), ID]
            nsample  = np.arange(0.,  len(traj)*self._dt, self._dt)
            nnsample = np.arange(0.,  len(traj)*self._dt-4*self._dt,  (len(traj)*self._dt-4*self._dt) / (1./self._dt))
            
            # print self._demo_traj.shape
            # print traj.shape
            # print len(self._time_stamps)
            # print len(nnsample)

            traj_data[:,ID+1] = interp1d(nsample, traj)(nnsample)
        
        return traj_data

    def create_kernel_fcn(self, centers, bfs_width):
        """
        this function creates a basis function. The basis fucntions are gaussians.
        The guassian is centered at a fixed centered (defined in the constructor)
        Args:
        centers : centers of the basis functions
        bfs_width : width of a basis function
        Returns:
        the kernel
        """
    
        def Kernel(q):
            """
            This function returns the computed kernel value
            Args: 
            q: the point for which kernel has to be computed
            Returns: 
            computed guassian point
            """
            D = np.sum( (np.diff(centers,1, axis=1)*0.55)**2, axis=0)
            D = 1./np.hstack([D, D[-1]])
            D = D * bfs_width
            
            res = np.zeros([centers.shape[1], q.shape[1]])
            for i in range(q.shape[1]):
                qq = npm.repmat( q[:,i], 1, centers.shape[1])
                res[:,i] = np.exp( -0.5* np.multiply( np.sum( (qq-centers)**2, axis=0), D))     
            
            return res
        y = Kernel
        
        return y

    def train(self):
        """
        function that trains the DMP object
        finds the weights of the DMP
        """
        #proportional gain
        K   = self._config['K']
        #derivative gain
        D   = self._config['D']
        #temporal scaling term
        tau = self._config['tau']
        #time constant of canonical system
        ax  = self._config['ax']

        #get the time stamp from assigned timestamp of the trajectory
        dt = self._time_stamps[2] - self._time_stamps[1]

        # canonical system roll out
        x = np.zeros([1, self._traj_data.shape[0]])
        x[0,0] = 1.
        # 1: Euler solution to exponential decreased canonical system
        for i in range(1, x.shape[1]):
            x[:,i] = x[:,i-1] + 1./tau * ax * x[:,i-1] * dt
  
        # calculate weights directly
        phi = self._kernel_fcn(x)
        w = np.zeros([phi.shape[0], self._traj_data.shape[1]-1])
        #normalizing denominator coefficeint
        deno = np.sum(phi, axis=1)[:,None]

        Y = self._traj_data[:, 1:]
        goals = Y[-1,:].T

        #first derivative of the trajectory
        Yd = np.vstack([np.zeros([1, Y.shape[1]]),  np.diff(Y, axis=0)/dt])
        #second derivative of the trjectory
        Ydd = np.vstack([np.zeros([1,Yd.shape[1]]), np.diff(Yd, axis=0)/dt])

        #find the weights for individual trajectories
        for i in range(1, self._traj_data.shape[1]):

            if  self._type == 1:
                y = -K * (goals[i-1] - Y[:,i-1]) + D * Yd[:,i-1] + tau * Ydd[:,i-1]
                if y.ndim == 1:
                    y = y[:,None]
            elif  self._type == 2 or self._type == 3:
                y = (tau * Ydd[:,i-1] + D * Yd[:,i-1])/K - (goals[i-1] - Y[:,i-1]) + (goals[i-1] - Y[1,i-1]) * x.T

            y = np.multiply(y, 1./(x.T))

            nume = np.dot(phi,y)
            wi = np.multiply(nume , 1./ (deno + 1e-6))

            w[:,i-1] = wi.squeeze()

        #assign the computed weights
        self._Ws = w

    def generate_trajectory(self, config=None):
        """
        this function is to generate a new DMP trajectory from already trained system
        Args: 
        config : contains all parameters like the input
        dt: time step
        K: porportional gain of the dmp
        D: derivative gain of the dmp
        y: the start location of the dmp
        dy the start velocity of the dmp
        goals: goals to which the dmp has to move - helps in spatial scaling
        tau: temporatl scale constant
        ext_force: optional parameter, denotes the external peturbation to a dmp, example, collision
        """

        if self._Ws is None:
            raise Exception("The DMP was not trained, call train method first before calling generate_trajectory")

        if config is None:
            config = self._config

        goals = config['goals']
        tau = config['tau']
        dt = config['dt']
        K = config['K']
        D = config['D']
        y = config['y0']
        dy = config['dy']

        if not ('ext_force' in  config.keys()):
            ext_force = np.array([0,0,0,0])
        else:
            ext_force = config['ext_force']

        u = 1.
        ax = config['ax']

        id = 1
        yreal = y
        dyreal = dy
        Y = yreal
        dY = dyreal
        ddY = np.zeros_like(dyreal)

        timestamps = np.array([0])
        t = 0
        #generate the trajectory
        #by rolling out out point at a time
        while u > 1e-3:

            id = id + 1
            kf = self._kernel_fcn(np.array([[u]]))

            forces = np.dot(self._Ws.T, kf/np.sum(kf))

            if  self._type == 1:
                scaling = np.multiply((goals - config['y0']), 1./config['original_scaling'])
                ddy = K * (goals - y) - D * dy + np.multiply(scaling, forces.T) * u
            elif  self._type == 2 or self._type == 3:
                scaling = goals - config['y0']
                ddy = K * (goals - y) - D * dy - K * scaling * u + K * forces.T * u
            
            # Euler Method, integration
            dy = dy + dt * ddy/tau
            y = y + dy * dt/tau

            if  self._type == 1 or self._type == 2:
                Y = np.vstack([Y,y])
                dY = np.vstack([dY, dy])
                ddY = np.vstack([ddY, ddy])
            
            #for the phase stop allowing type of DMP
            elif  self._type == 3:
                Ky = 300.
                Dy = np.sqrt(4*Ky)
                ddyreal = Ky * (y - yreal) - Dy * dyreal
                if (timestamps[id-1] >= ext_force[1]) and (timestamps[id-1] < ext_force[1] + ext_force[2]):
                    ddyreal = ddyreal + ext_force[3:]
                
                dyreal = dyreal + ddyreal * dt
                yreal = yreal + dyreal * dt
                Y = np.vstack([Y,yreal])
                dY = np.vstack([dY, dyreal])
                ddY = np.vstack([ddY, ddy])
            
            #canonical system rollout based on the typpe of the system
            if  self._type == 1 or self._type == 2:
                u = u + 1/tau * ax * u * dt
            elif  self._type == 3:
                phasestop = 1 + config['ac'] * np.sqrt(np.sum((yreal - y)**2))
                u = u + 1/tau * ax * u * dt / phasestop

            t = t + dt
            timestamps = np.hstack([timestamps, t])

        #append the trajectory
        traj   = np.hstack([np.asarray(timestamps)[:,None], Y])
        dtraj  = np.hstack([np.asarray(timestamps)[:,None], dY])
        ddtraj = np.hstack([np.asarray(timestamps)[:,None], ddY])

        traj_data = {
        #computed traj of the trajectory
        'pos':traj,
        #computed vel of the trajectory
        'vel':dtraj,
        #computed acc of the trajectory
        'acc':ddtraj,
        }

        return traj_data