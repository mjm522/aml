import numpy as np
import scipy.interpolate
from utilities import make_demonstrations

class DiscreteDMPShell(): 
    def __init__(self, params):

        self._n_bfs       = params['n_bfs'] #number of basis functions
        self._tau          = params['duration'] #duration of demonstration

        self._y0           = params['start'] #start position
        self._goal         = params['goal'] #goal position

        if params['path'] is None:
            self._des_path = make_demonstrations(start=self._y0, goal=self._goal)
        else:
            self._des_path = params['path']

        self._dt           = params['dt']
        self._run_time     = 1.0
        self._timesteps    = int(self._run_time / self._dt)
        
        self._weights      = np.zeros([1, self._n_bfs]) # DMP Gaussian kernel weights

        self._alpha_z      = 25. # Schaal 2012
        self._beta_z       = self._alpha_z / 4. # Schaal 2012
        
        #generate centers for the canonical system
        self._ax          = 1.0
        self._cs_centers   = self.gen_cs_centers()
        
        # set variance of Gaussian basis functions
        self._sigma        = np.ones(self._n_bfs) * self._n_bfs**1.5 / self._cs_centers

        self._true_traj    = self.gen_path(self._des_path)
        
        self.check_offset()
        self.reset_state_cs()
        # set up the DMP system
        self.reset_state_dmp()

    def gen_cs_centers(self):
        """Set the centre of the Gaussian basis 
        functions be spaced evenly throughout run time"""
        # desired spacings along x
        # need to be spaced evenly between 1 and exp(-ax)
        # lowest number should be only as far as x gets 
        first = np.exp(-self._ax*self._run_time) 
        last = 1.05 - first
        des_c = np.linspace(first,last,self._n_bfs) 

        cs_centers = np.ones(len(des_c)) 
        for n in range(len(des_c)): 
            cs_centers[n] = -np.log(des_c[n])  # x = exp(-c), solving for c
        return cs_centers

    def gen_path(self, trajectory):
        """Generate the DMPs necessary to follow the 
        specified trajectory.

        trajectory np.array: the time series of points to follow
                             [DOFs, time], with a column of None
                             wherever the pen should be lifted
        """
        return self.imitate_path(y_des=trajectory)


    def imitate_path(self, y_des):
        """Takes in a desired trajectory and generates the set of 
        system parameters that best realize this path.
    
        y_des list/array: the desired trajectories of each DMP should be shaped [1, run_time]
        """
        # set initial state and goal
        if y_des.ndim == 1: 
            y_des = y_des.reshape(1,len(y_des))
        self._y0 = y_des[0,0].copy()
        self.y_des = y_des.copy()
        self._goal = self.gen_goal(y_des)
        
        self.check_offset()
        
        # generate function to interpolate the desired trajectory
        path = np.zeros([1, self._timesteps])
        x = np.linspace(0, self._run_time, y_des.shape[1])

        path_gen = scipy.interpolate.interp1d(x, y_des)
        #print x.shape
        for t in range(self._timesteps):  
            path[0, t] = path_gen(t * self._dt)
        #here basically the path interpolated became a trajectory. The number of points
        #given will have to be achieved in cs.run_time
        y_des = path
        #print path.shape
        # calculate velocity of y_des
        dy_des = np.diff(y_des) / self._dt
        # add zero to the beginning of every row
        dy_des = np.hstack([np.zeros([1,1]), dy_des])
        # calculate acceleration of y_des
        ddy_des = np.diff(dy_des) / self._dt
        # add zero to the beginning of every row
        ddy_des = np.hstack([np.zeros([1,1]), ddy_des])
        f_target = np.zeros([y_des.shape[1], 1])
        
        # find the force required to move along this trajectory
        f_target = ddy_des - self._alpha_z * (self._beta_z * (self._goal - y_des) - dy_des)
            #f_target[:,d]   =   (ddy_des[d]+dy_des[d])/(self._alpha_z[d]*self._beta_z[d]) - (self._goal[d] - y_des[d]) + (self._goal[d]-self._y0[d]);

        # efficiently generate weights to realize f_target
        self.gen_weights(f_target)
        self.reset_state_dmp()

        return y_des

    def check_offset(self):
        """Check to see if initial position and goal are the same
        if they are, offset slightly so that the forcing term is not 0"""
        if self._y0 == self._goal:
            self._goal += 1e-4

    def gen_weights(self, f_target):
        """Generate a set of weights over the basis functions such 
        that the target forcing term trajectory is matched.
        
        f_target np.array: the desired forcing term trajectory
        """
        # calculate x and psi   
        x_track = self.rollout_cs()

        psi_track = self.gen_psi(x_track)

        #efficiently calculate weights for BFs using weighted linear regression
        self._weights = np.zeros([1, self._n_bfs])
        # spatial scaling term
        k = (self._goal - self._y0)
        for b in range(self._n_bfs):
            numer = np.sum(x_track * psi_track[:,b] * f_target)
            denom = np.sum(x_track**2 * psi_track[:,b])
            self._weights[0, b] = numer / (k * denom)

    def set_goal(self, goal):
        self._goal = goal

    def reset_state_dmp(self, new_start=None):
        """Reset the system state"""
        if new_start is None:
            self.y = self._y0.copy()
        else:
            self.y = new_start
        self.dy = np.zeros(1)   
        self.ddy = np.zeros(1)  
        self.reset_state_cs()

    def reset_state_cs(self):
        """Reset the system state"""
        self.x = 1.0


    def gen_psi(self, x):
        """Generates the activity of the basis functions for a given 
        canonical system rollout. 
        
        x float, array: the canonical system state or path
        """
        if isinstance(x, np.ndarray):
            x = x[:,None]
        df = (x - self._cs_centers)**2;

        return np.exp(-self._sigma * (x - self._cs_centers)**2)


    def gen_front_term(self, x):
        """Generates the diminishing front term on 
        the forcing term.

        x float: the current value of the canonical system
        dmp_num int: the index of the current dmp
        """
        return x * (self._goal - self._y0)

    def step_dmp(self, tau=None, state_fb=None, external_force=None):
        """Run the DMP system for a single timestep.

       tau float: scales the timestep
                  increase tau to make the system execute faster
       state_fb np.array: optional system feedback
        """
        # run canonical system
        if tau is None:
            tau = 1.0
        cs_args = {'tau':tau,
                   'error_coupling':1.0}
        if state_fb is not None: 
            # take the 2 norm of the overall error
            state_fb = state_fb.reshape(1,self.num_dmps)
            dist = np.sqrt(np.sum((state_fb - self.y)**2))
            cs_args['error_coupling'] = 1.0 / (1.0 + 10*dist)
        
        x = self.step_cs(**cs_args)

        # generate basis function activation
        psi = self.gen_psi(x)[:,None]

        # generate the forcing term
        f = (self.gen_front_term(x) * (np.dot(self._weights, psi)) / np.sum(psi))[0]

        # DMP acceleration
        self.ddy = (self._alpha_z * (self._beta_z * (self._goal - self.y) - self.dy/tau) + f) * tau
        if external_force is not None:
            self.ddy += external_force
        
        #integrating the system
        self.dy += self.ddy * tau * self._dt * cs_args['error_coupling']
        self.y += self.dy * self._dt * cs_args['error_coupling']

        return self.y, self.dy, self.ddy, psi

    def rollout_dmp(self, tau=None):
        """Generate a system trial, no feedback is incorporated."""
        #self.reset_state_dmp()

        if tau is None:
            timesteps = self._timesteps
        else:
            timesteps = int(self._timesteps/tau)
 
        # set up tracking vectors
        y_track   = np.zeros([timesteps, 1]) 
        dy_track  = np.zeros([timesteps, 1])
        ddy_track = np.zeros([timesteps, 1])
    
        for t in range(timesteps):
            y, dy, ddy, _ = self.step_dmp(tau)
            # record timestep
            y_track[t,:] = y
            dy_track[t,:] = dy
            ddy_track[t,:] = ddy

        return y_track, dy_track, ddy_track

    def gen_goal(self, y_des): 
        """Generate the goal for path imitation. 
        For rhythmic DMPs the goal is the average of the 
        desired trajectory.
    
        y_des np.array: the desired trajectory to follow
        """
        return y_des[:,-1].copy()

    def step_cs(self, tau=1.0, error_coupling=1.0):
        """Generate a single step of x for discrete
        (potentially closed) loop movements. 
        Decaying from 1 to 0 according to dx = -ax*x.
        
        tau float: gain on execution time
                   increase tau to make the system execute faster
        error_coupling float: slow down if the error is > 1
        """
        self.x += (-self._ax* self.x * error_coupling) * tau * self._dt
        return self.x

    def rollout_cs(self, **kwargs):
        """Generate x for open loop movements.
        """
        if kwargs.has_key('tau'):
            timesteps = int(self._timesteps / kwargs['tau'])
        else: 
            timesteps = self._timesteps
        self.x_track = np.zeros(timesteps)
        
        self.reset_state_cs()
        for t in range(timesteps):
            self.x_track[t] = self.x 
            self.step_cs(**kwargs)

        return self.x_track
