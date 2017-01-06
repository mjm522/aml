import numpy as np
import quaternion
import copy

from aml_lfd.utilities.utilities import compute_w

class MinJerkInterp():
    def __init__(self, dt=0.05, tau=5.):
        self.dt = dt
        self.tau = tau
        self.timesteps = np.arange(0, 2*self.tau, self.dt)
        self.min_jerk_traj = {}

    def configure(self, start_pos, start_qt, goal_pos, goal_qt):
        self.start_pos = copy.deepcopy(start_pos)
        self.goal_pos  = copy.deepcopy(goal_pos)
        if isinstance(start_qt, np.quaternion):
            start_qt = quaternion.as_float_array(start_qt)[0]
        if isinstance(goal_qt, np.quaternion):
            goal_qt = quaternion.as_float_array(goal_qt)[0]
        self.start_qt  = copy.deepcopy(start_qt)
        self.goal_qt   = copy.deepcopy(goal_qt)

    def min_jerk_step(self, x, xd, xdd, goal, tau):
        # function [x,xd,xdd] = min_jerk_step(x,xd,xdd,goal,tau, dt) computes
        # the update of x,xd,xdd for the next time step dt given that we are
        # currently at x,xd,xdd, and that we have tau until we want to reach
        # the goal

        if tau < self.dt:
        	return np.nan, np.nan, np.nan

        dist = goal - x

        a1   = 0
        a0   = xdd * tau**2
        v1   = 0
        v0   = xd * tau

        t1 = self.dt
        t2 = self.dt**2
        t3 = self.dt**3
        t4 = self.dt**4
        t5 = self.dt**5

        c1 = (6.*dist + (a1 - a0)/2. - 3.*(v0 + v1))/(tau**5)
        c2 = (-15.*dist + (3.*a0 - 2.*a1)/2. + 8.*v0 + 7.*v1)/(tau**4)
        c3 = (10.*dist+ (a1 - 3.*a0)/2. - 6.*v0 - 4.*v1)/(tau**3)
        c4 = xdd/2.
        c5 = xd
        c6 = x

        x   = c1*t5 + c2*t4 + c3*t3 + c4*t2 + c5*t1 + c6
        xd  = 5.*c1*t4 + 4.*c2*t3 + 3.*c3*t2 + 2.*c4*t1 + c5
        xdd = 20.*c1*t3 + 12.*c2*t2 + 6.*c3*t1 + 2.*c4

        if np.isnan(x) or np.isnan(xd) or np.isnan(xdd):
            
            x   = 0.
            xd  = 0.
            xdd = 0.

        return x, xd, xdd


    def min_jerk_step_pos(self):

        final_p = np.zeros((len(self.timesteps),3))

        for j in range(3):
            # generate the minimum jerk trajectory between each component of position
            t    = self.start_pos[j]
            td   = 0
            tdd  = 0
            goal = self.goal_pos[j]
            T    = np.zeros((len(self.timesteps),3))
            for i in range(len(self.timesteps)):
            	t,td,tdd = self.min_jerk_step( t, td, tdd, goal, self.tau-i*self.dt)
            	T[i,:]   = np.array([t, td, tdd])
              	#print i, '\t', T[i,:]
            #print T[:,j]
            final_p[:,j] = T[:,0].copy()

        #differentiate
        final_v = np.diff(final_p, axis=0)/self.dt
        #add initial velocity
        final_v = np.vstack([np.zeros((1,3)),final_v])

        #compute angular acceleration
        final_a = np.diff(final_v, axis=0)/self.dt
        #add initial acceleration
        final_a = np.vstack([np.zeros((1,3)),final_a])

        return final_p, final_v, final_a


    def  min_jerk_step_qt(self):
        final_q = np.zeros((len(self.timesteps),4))

        for j in range(4):
            # generate the minimum jerk trajectory between each component of
            # quarternions
            t = self.start_qt[j]
            td = 0
            tdd = 0
            goal = self.goal_qt[j]
            T = np.zeros((len(self.timesteps),3))
            for i in range(len(self.timesteps)):
                t,td,tdd = self.min_jerk_step(t, td, tdd, goal, self.tau-i*self.dt)
                T[i,:]   = np.array([t, td, tdd])

            final_q[:,j] = T[:,0].copy()

        #normalize the quarternions
        for i in range(len(self.timesteps)):
            tmp = final_q[i,:]
            final_q[i,:] = tmp/np.linalg.norm(tmp)

        #differentiate
        final_dot = np.diff(final_q, axis=0)/self.dt
        #add initial velocity
        final_dot = np.vstack([np.zeros((1,4)),final_dot])

        #compute angular velocity
        final_w = np.zeros((len(self.timesteps),3))
        for i in range(len(self.timesteps)):
            final_w[i,:] = compute_w(final_q[i,:], final_dot[i,:])

        #compute angular acceleration
        final_al = np.diff(final_w, axis=0)/self.dt

        #add initial acceleration
        final_al = np.vstack([np.zeros((1,3)),final_al])

        #converting to a quaternion array
        final_q = quaternion.as_quat_array(final_q)

        return final_q, final_w, final_al

    def get_interpolated_trajectory(self):

        final_q, final_w, final_al  = self.min_jerk_step_qt()

        final_p, final_v, final_a   = self.min_jerk_step_pos()
    
        #position trajectory
        self.min_jerk_traj['pos_traj'] = final_p 
        #velocity trajectory
        self.min_jerk_traj['vel_traj'] = final_v
        #acceleration trajectory
        self.min_jerk_traj['acc_traj'] = final_al


        #orientation trajectory
        self.min_jerk_traj['ori_traj'] = final_q
        #angular velocity trajector
        self.min_jerk_traj['omg_traj'] = final_w
        #angular acceleration trajectory
        self.min_jerk_traj['ang_traj'] = final_a

        return self.min_jerk_traj

    def plot_min_jerk_traj(self):
        min_jerk_traj = self.get_interpolated_trajectory()

        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.subplot(311)
        #for plotting this return orientation interpolation as a numpy array
        # plt.title('orientation')
        # plt.plot(min_jerk_traj['ori_traj'][:,0]) 
        # plt.plot(min_jerk_traj['ori_traj'][:,1]) 
        # plt.plot(min_jerk_traj['ori_traj'][:,2]) 
        # plt.plot(min_jerk_traj['ori_traj'][:,3])
        # 
        plt.subplot(312)
        plt.title('omega')
        plt.plot(min_jerk_traj['omg_traj'][:,0]) 
        plt.plot(min_jerk_traj['omg_traj'][:,1]) 
        plt.plot(min_jerk_traj['omg_traj'][:,2]) 
        # 
    
        plt.subplot(313)
        plt.title('alpha')
        plt.plot(min_jerk_traj['ang_traj'][:,0]) 
        plt.plot(min_jerk_traj['ang_traj'][:,1]) 
        plt.plot(min_jerk_traj['ang_traj'][:,2])

        plt.figure(2)
        plt.subplot(311)
        plt.title('position')
        plt.plot(min_jerk_traj['pos_traj'][:,0]) 
        plt.plot(min_jerk_traj['pos_traj'][:,1]) 
        plt.plot(min_jerk_traj['pos_traj'][:,2]) 
        # 
        plt.subplot(312)
        plt.title('velocity')
        plt.plot(min_jerk_traj['vel_traj'][:,0]) 
        plt.plot(min_jerk_traj['vel_traj'][:,1]) 
        plt.plot(min_jerk_traj['vel_traj'][:,2]) 
        # 
        plt.subplot(313)
        plt.title('acceleration')
        plt.plot(min_jerk_traj['acc_traj'][:,0]) 
        plt.plot(min_jerk_traj['acc_traj'][:,1]) 
        plt.plot(min_jerk_traj['acc_traj'][:,2])

        plt.show()


#test code
def main():
    minjerk   = MinJerkInterp()
    start_pos = np.array([1.,2.,3.])
    goal_pos  = np.array([2.,3.,4.])
    start_qt  = np.quaternion(1.0, 0.,0., 0.)
    goal_qt   = np.quaternion(0.707, 0.707,0.,0.)
    minjerk.configure(start_pos=start_pos, goal_pos=goal_pos, start_qt=start_qt, goal_qt=goal_qt)
    minjerk.plot_min_jerk_traj()


if __name__ == '__main__':
    main()