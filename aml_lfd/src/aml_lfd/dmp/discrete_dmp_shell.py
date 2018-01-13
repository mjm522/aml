import numpy as np
import numpy.matlib as npm
# from aml_lfd.lfd import LfD
from scipy.interpolate import interp1d


class DiscreteDMPShell(object):

    def __init__(self, config):
        
        # LfD.__init__(self, config)
        self._config = config
        # self._end_time = config['end_time']
        self._dt = config['dt']
        
        dc = 1./(config['rbf_num'] -1 )
        centers = np.arange(1., 0., -dc)[None, :]
        self._kernel_fcn = self.create_kernel_fcn(centers, 1)
        self._time_stamps = np.arange(0, config['end_time'], self._dt)

        self._type = config['type']
        self._dof = config['dof']

        self._traj_data = None
        self._Ws = None

    def load_demo_trajectory(self, trajectory):
        if trajectory.shape[1] != self._dof:
            raise("The trajectory does not have same number of dimensions as specified")
        self._demo_traj = np.vstack([trajectory, npm.repmat(trajectory[-1,:],20,1)])
        self._traj_data = self.process_trajectory()
        self._config['original_scaling'] = self._traj_data[-1,1:] - self._traj_data[1,1:] + 1e-5

    def get_time_stamps(self):
        return np.arange(0, self._end_time, self._dt)


    def process_trajectory(self):
        
        traj_data = np.zeros([self._time_stamps.shape[0], self._dof+1])
        traj_data[:,0] = self._time_stamps
        
        for ID in range(self._dof):

            traj = self._demo_traj[:,ID]
            nsample = np.arange(0.,  len(traj)*self._dt, self._dt)
            nnsample = np.arange(0., len(traj)*self._dt-4*self._dt,  (len(traj)*self._dt-4*self._dt) / (1./self._dt))
            traj_data[:,ID+1] = interp1d(nsample, traj)(nnsample)
        
        return traj_data

    def create_kernel_fcn(self, centers, dd):
    
        def Kernel(q):
            D = np.sum( (np.diff(centers,1, axis=1)*0.55)**2, axis=0)
            D = 1./np.hstack([D, D[-1]])
            D = D * dd
            
            res = np.zeros([centers.shape[1], q.shape[1]])
            for i in range(q.shape[1]):
                qq = npm.repmat( q[:,i], 1, centers.shape[1])
                res[:,i] = np.exp( -0.5* np.multiply( np.sum( (qq-centers)**2, axis=0), D))     
            
            return res
        y = Kernel
        
        return y

    def train(self):

        K   = self._config['K']
        D   = self._config['D']
        tau = self._config['tau']
        ax  = self._config['ax']

        dt = self._time_stamps[2] - self._time_stamps[1]

        # canonical system
        x = np.zeros([1, self._traj_data.shape[0]])
        x[0,0] = 1.
        # 1: Euler solution to exponential decreased canonical system
        for i in range(1, x.shape[1]):
            x[:,i] = x[:,i-1] + 1./tau * ax * x[:,i-1] * dt
  
        # calculate weights directly
        phi = self._kernel_fcn(x)
        w = np.zeros([phi.shape[0], self._traj_data.shape[1]-1])
        deno = np.sum(phi, axis=1)[:,None]

        Y = self._traj_data[:, 1:]
        goals = Y[-1,:].T

        Yd = np.vstack([np.zeros([1, Y.shape[1]]),  np.diff(Y, axis=0)/dt])
        Ydd = np.vstack([np.zeros([1,Yd.shape[1]]), np.diff(Yd, axis=0)/dt])

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

        self._Ws = w

    def test(self, config=None):

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
        timestamps = np.array([0])
        t = 0
        while u > 1e-3:
            print u
            raw_input()
            id = id + 1
            kf = self._kernel_fcn(np.array([[u]]))

            forces = np.dot(self._Ws.T, kf/np.sum(kf))

            if  self._type == 1:
                scaling = np.multiply((goals - config['y0']), 1./config['original_scaling'])
                ddy = K * (goals - y) - D * dy + np.multiply(scaling, forces.T) * u
            elif  self._type == 2 or self._type == 3:
                scaling = goals - config['y0']
                ddy = K * (goals - y) - D * dy - K * scaling * u + K * forces.T * u
            
            # Euler Method
            dy = dy + dt * ddy/tau
            y = y + dy * dt/tau

            if  self._type == 1 or self._type == 2:
                Y = np.vstack([Y,y])
            elif  self._type == 3:
                Ky = 300.
                Dy = np.sqrt(4*Ky)
                ddyreal = Ky * (y - yreal) - Dy * dyreal
                if (timestamps[id-1] >= ext_force[1]) and (timestamps[id-1] < ext_force[1] + ext_force[2]):
                    ddyreal = ddyreal + ext_force[3:]
                
                dyreal = dyreal + ddyreal * dt
                yreal = yreal + dyreal * dt
                Y = np.vstack([Y,yreal])
            
            #canonical system
            if  self._type == 1 or self._type == 2:
                u = u + 1/tau * ax * u * dt
            elif  self._type == 3:
                phasestop = 1 + config['ac'] * np.sqrt(np.sum((yreal - y)**2))
                u = u + 1/tau * ax * u * dt / phasestop

            t = t + dt
            timestamps = np.hstack([timestamps, t])

        traj = np.hstack([np.asarray(timestamps)[:,None], Y])

        return traj