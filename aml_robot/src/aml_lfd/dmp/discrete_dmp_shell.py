import numpy as np
from aml_lfd.lfd import LfD
from config import DISCRETE_DMP

class DiscreteDMPShell(LfD):
    """
    A shell that uses dynamic movement primitives to 
    control a robotic arm end-effector.
    """   
    def __init__(self, config = DISCRETE_DMP):
        """
        bfs int: the number of basis functions per DMP
        add_to_goals np.array: floats to add to the DMP goals
                               used to scale the DMPs spatially
        pattern string: specifies either 'discrete' or 'rhythmic' DMPs
        """
        LfD.__init__(self, config)

        self._num_dmps     = config['dmps']
        self._bfs          = config['bfs']

        self._gain         = config['gain']
        self._tau          = config['tau']
        
        self._threshold    = config['threshold']

        self._pen_down     = config['pen_down']
        self._dt           = config['dt']

        self._des_path     = None
        self._y0           = None
        self._goal         = None
         
        self._done          = False
        self._not_at_start  = True
        self._num_seq       = config['num_seq']
        self._run_time      = config['run_time']
        
        self._timesteps    = None
        
        self._weights      = np.zeros((self._num_dmps, self._bfs))

        self._alpha_z      = np.ones(self._num_dmps) * config['alpha_z'] # Schaal 2012
        self._beta_z       = self._alpha_z.copy() / config['beta_z'] # Schaal 2012
        
        self._ax           = 1.0

        #generate centers for the canonical system
        self._cs_centers   = None
        # set variance of Gaussian basis functions
        self._sigma        = None

        self._true_traj    = None

    def encode_demo(self):
        pass
        
    def configure(self, traj2follow, start, goal):
    	self._des_path    = traj2follow
    	self._y0          = start
        self._goal        = goal 
                
        self._timesteps   = int(self._run_time / self._dt)

        self._cs_centers  = self.gen_cs_centers()

        self._sigma       = np.ones(self._bfs) * self._bfs**1.5 / self._cs_centers

        self._true_traj   = self.gen_path(self._des_path)

        # self.set_target()
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
        des_c = np.linspace(first, last, self._bfs) 

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
        # break up the trajectory into its different words
        # NaN or None signals a new word / break in drawing
        breaks = np.array(np.where(trajectory[0] != trajectory[0]))[0] 
        self._num_seqs = len(breaks) - 1       
        self._dmp_sets = []
        return self.imitate_path(y_des=trajectory)


    def imitate_path(self, y_des):
        """Takes in a desired trajectory and generates the set of 
        system parameters that best realize this path.
    
        y_des list/array: the desired trajectories of each DMP
                          should be shaped [dmps, run_time]
        """
        # set initial state and goal
        if y_des.ndim == 1: 
            y_des = y_des.reshape(1,len(y_des))

        self._y0 = y_des[:,0].copy()
        self._y_des = y_des.copy()
        self._goal = self.gen_goal(y_des)
        
        self.check_offset()
        
        # generate function to interpolate the desired trajectory
        import scipy.interpolate
        path = np.zeros((self._num_dmps, self._timesteps))
        x = np.linspace(0, self._run_time, y_des.shape[1])
        for d in range(self._num_dmps):
            path_gen = scipy.interpolate.interp1d(x, y_des[d])
            #print x.shape
            for t in range(self._timesteps):  
                path[d, t] = path_gen(t * self._dt)
        #here basically the path interpolated became a trajectory. The number of points
        #given will have to be achieved in cs.run_time
        y_des = path
        #print path.shape
        # calculate velocity of y_des
        dy_des = np.diff(y_des) / self._dt
        # add zero to the beginning of every row
        dy_des = np.hstack((np.zeros((self._num_dmps, 1)), dy_des))
        # calculate acceleration of y_des
        ddy_des = np.diff(dy_des) / self._dt
        # add zero to the beginning of every row
        ddy_des = np.hstack((np.zeros((self._num_dmps, 1)), ddy_des))
        f_target = np.zeros((y_des.shape[1], self._num_dmps))
        # find the force required to move along this trajectory
        for d in range(self._num_dmps):
            f_target[:,d] = ddy_des[d] - self._alpha_z[d] * (self._beta_z[d] * (self._goal[d] - y_des[d]) - dy_des[d])
            #f_target[:,d]   =   (ddy_des[d]+dy_des[d])/(self.alpha_z[d]*self.beta_z[d]) - (self.goal[d] - y_des[d]) + (self.goal[d]-self.y0[d]);

        # efficiently generate weights to realize f_target
        self.gen_weights(f_target)

        self.reset_state_dmp()
        return y_des

    def check_offset(self):
        """Check to see if initial position and goal are the same
        if they are, offset slightly so that the forcing term is not 0"""

        for d in range(self._num_dmps):
            if (self._y0[d] == self._goal[d]):
                self._goal[d] += 1e-4

    def gen_weights(self, f_target):
        """Generate a set of weights over the basis functions such 
        that the target forcing term trajectory is matched.
        
        f_target np.array: the desired forcing term trajectory
        """
        # calculate x and psi   
        x_track = self.rollout_cs()

        psi_track = self.gen_psi(x_track)

        #efficiently calculate weights for BFs using weighted linear regression
        self._weights = np.zeros((self._num_dmps, self._bfs))
        for d in range(self._num_dmps):
            # spatial scaling term
            k = (self._goal[d] - self._y0[d])
            for b in range(self._bfs):
                numer = np.sum(x_track * psi_track[:,b] * f_target[:,d])
                denom = np.sum(x_track**2 * psi_track[:,b])
                self._weights[d,b] = numer / (k * denom)

    def reset_state_dmp(self, new_start=None):
        """Reset the system state"""
        if new_start is None:
            self._y = self._y0.copy()
        else:
            self._y = new_start
        self._dy = np.zeros(self._num_dmps)   
        self._ddy = np.zeros(self._num_dmps)  
        self.reset_state_cs()

    def reset_state_cs(self):
        """Reset the system state"""
        self._x = 1.0


    def gen_psi(self, x):
        """Generates the activity of the basis functions for a given 
        canonical system rollout. 
        
        x float, array: the canonical system state or path
        """
        if isinstance(x, np.ndarray):
            x = x[:,None]
        df = (x - self._cs_centers)**2;

        return np.exp(-self._sigma * (x - self._cs_centers)**2)


    def gen_front_term(self, x, dmp_num):
        """Generates the diminishing front term on 
        the forcing term.

        x float: the current value of the canonical system
        dmp_num int: the index of the current dmp
        """
        return x * (self._goal[dmp_num] - self._y0[dmp_num])

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
        psi = self.gen_psi(x)

        for d in range(self._num_dmps):
            # generate the forcing term
            f = self.gen_front_term(x, d) * \
                (np.dot(psi, self._weights[d])) / np.sum(psi)
            # DMP acceleration
            self._ddy[d] = (self._alpha_z[d] * (self._beta_z[d] * (self._goal[d] - self._y[d]) - self._dy[d]/tau) + f) * tau
            if external_force is not None:
                self._ddy[d] += external_force[d]
            #integrating the system
            self._dy[d] += self._ddy[d] * tau * self._dt * cs_args['error_coupling']
            self._y[d] += self._dy[d] * self._dt * cs_args['error_coupling']

        return self._y, self._dy, self._ddy

    def rollout_dmp(self, tau=None):
        """Generate a system trial, no feedback is incorporated."""
        #self.reset_state_dmp()

        if tau is None:
            timesteps = self._timesteps
        else:
            timesteps = int(self._timesteps/tau)
 
        # set up tracking vectors
        y_track = np.zeros((timesteps, self._num_dmps)) 
        dy_track = np.zeros((timesteps, self._num_dmps))
        ddy_track = np.zeros((timesteps, self._num_dmps))
    
        for t in range(timesteps):
            y, dy, ddy = self.step_dmp(tau)
            # record timestep
            y_track[t] = y
            dy_track[t] = dy
            ddy_track[t] = ddy

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
        self._x += (-self._ax * self._x * error_coupling) * tau * self._dt
        return self._x

    def rollout_cs(self, **kwargs):
        """Generate x for open loop movements.
        """
        if kwargs.has_key('tau'):
            timesteps = int(self._timesteps / kwargs['tau'])
        else: 
            timesteps = self._timesteps
        self._x_track = np.zeros(timesteps)
        
        self.reset_state_cs()
        for t in range(timesteps):
            self._x_track[t] = self._x 
            self.step_cs(**kwargs)

        return self._x_track

    def set_next_seq(self):
        """Get the next sequence in the list.
        """
        self._num_dmps = self.dmp_sets[self._num_seq]