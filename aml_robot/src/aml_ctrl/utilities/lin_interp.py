import numpy as np
import quaternion

from aml_lfd.utilities.utilities import quat_lerp, compute_log, quat_mult, quat_conj

class LinInterp():
    def __init__(self, dt=0.05, tau=5.):
        self.dt = dt
        self.tau = tau
        self.timesteps = np.arange(0, 2*self.tau, self.dt)
        self.lin_interp_traj = {}

    def configure(self, start_pos, start_qt, goal_pos, goal_qt):
        self.start_pos = start_pos
        self.goal_pos  = goal_pos
        if isinstance(start_qt, np.quaternion):
            start_qt = quaternion.as_float_array(start_qt)[0]
        if isinstance(goal_qt, np.quaternion):
            goal_qt = quaternion.as_float_array(goal_qt)[0]
        self.start_qt  = start_qt
        self.goal_qt   = goal_qt

    def lin_step_qt(self):
        
        n_steps = len(self.timesteps)

        final_q  = np.zeros((n_steps,4))
        final_w  = np.zeros((n_steps,3))

        for t in range(n_steps):
            final_q[t,:]   = quat_lerp(self.start_qt, self.goal_qt, t/n_steps)
            
            #compute angular velocity
            if t < n_steps-1:        
                final_w[t+1,:] = 2.*compute_log(quat_mult(self.goal_qt, quat_conj(final_q[t,:])))

        #compute angular acceleration
        final_al = np.diff(final_w, axis=0)/self.dt

        #add initial acceleration
        final_al = np.vstack([np.zeros((1,3)),final_al])

        #converting to a quaternion array
        final_q = quaternion.as_quat_array(final_q)

        return final_q, final_w, final_al

    def lin_step_pos(self):
        n_steps = len(self.timesteps)
        
        final_p   = np.vstack([np.linspace(self.start_pos[0], self.goal_pos[0], n_steps),
                               np.linspace(self.start_pos[1], self.goal_pos[1], n_steps),
                               np.linspace(self.start_pos[2], self.goal_pos[2], n_steps)]).T

        #differentiate
        final_v = np.diff(final_p, axis=0)/self.dt
        #add initial velocity
        final_v = np.vstack([np.zeros((1,3)),final_v])

        #compute angular acceleration
        final_a = np.diff(final_v, axis=0)/self.dt
        #add initial acceleration
        final_a = np.vstack([np.zeros((1,3)),final_a])

        return final_p, final_v, final_a

    def get_interpolated_trajectory(self):

        final_q, final_w, final_al  = self.lin_step_qt()

        final_p, final_v, final_a   = self.lin_step_pos()

        #position trajectory
        self.lin_interp_traj['pos_traj'] = final_p
        #velocity trajectory
        self.lin_interp_traj['vel_traj'] = final_v
        #acceleration trajectory
        self.lin_interp_traj['acc_traj'] = final_al
        #orientation trajectory
        self.lin_interp_traj['ori_traj'] = final_q
        #angular velocity trajector
        self.lin_interp_traj['omg_traj'] = final_w
        #angular acceleration trajectory
        self.lin_interp_traj['alp_traj'] = final_a

        return self.lin_interp_traj

    def plot_lin_interp_traj(self):
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
        plt.plot(min_jerk_traj['alp_traj'][:,0]) 
        plt.plot(min_jerk_traj['alp_traj'][:,1]) 
        plt.plot(min_jerk_traj['alp_traj'][:,2])

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
    minjerk   = LinInterp()
    start_pos = np.array([1.,2.,3.])
    goal_pos  = np.array([2.,3.,4.])
    start_qt  = np.quaternion(1.0, 0.,0., 0.)
    goal_qt   = np.quaternion(0.707, 0.707,0.,0.)
    minjerk.configure(start_pos=start_pos, goal_pos=goal_pos, start_qt=start_qt, goal_qt=goal_qt)
    minjerk.plot_lin_interp_traj()


if __name__ == '__main__':
    main()


