import numpy as np
import matplotlib.pyplot as plt
from aml_opt.pi_cmd_opt.pi_cmd_opt import PICmdOpt


ctrl_constraints={
'min':[-4.],
'max':[4.],
}

config = {
    'max_iterations':2000,
    'dt': 0.01,
    'N':15,
    'h':10.,
    'rho': 1.5,#10.7,#0.2
    'K':10,
    'cmd_dim': 1,
    'state_dim': 2,
    'random_seed': 123,
    'init_policy': 'use_random',
    'ctrl_constraints':ctrl_constraints,
    }

class OptimCmd(PICmdOpt):

    def __init__(self, config):

        PICmdOpt.__init__(self, config)

    def dynamics(self, x, u, dt=0.01):

        A = np.array([[0.,1.],[1., 0.]])
        B = np.array([[0.],[1.]])

        x_nxt = np.dot( (np.eye(2) + A*dt), x) + np.dot(B, u)

        return x_nxt

    def cost(self, x, u, du, t):

        x_goal = np.array([1.,1])

        return 10.*np.dot( (x_goal-x),(x_goal-x) ) + 0.5*np.dot(u,u)


    def forward_rollout(self, u_list, k):

        N = u_list.shape[1]

        xs_samples = np.zeros((self._N, self._state_dim)) # Forward trajectory samples
        delus_samples = np.zeros((self._N, self._cmd_dim))
        
        ss = np.zeros(self._N) # Costs

        xs_samples[0,:] = self._x0.copy()

        for t in range(0, self._N):
            
            x_prev = xs_samples[t,:]
            
            du = self.calc_delu(t)

            x_next  = self.dynamics(x_prev, u_list[:,t] + du)

            xs_samples[t,:] = x_next

            ss[t] += ss[t] + self.cost(x_next, u_list[:,t], du, t+1)
            
            if ss[t] < 1e-4 or np.isnan(ss[t]):
                ss[t] = 0.
            
            delus_samples[t,:] = du

        return xs_samples, ss, delus_samples


def main():

    pi_cmd = OptimCmd(config=config)

    u_list = np.random.randn(config['cmd_dim'], config['N'])

    u_list, xs_samples, traj_prob, delus_samples = pi_cmd.run(u_list)

    x_final = np.zeros([config['N']+1, config['state_dim']])

    x_final[0, :] = pi_cmd._x0.copy()

    for k in range(config['N']):
        x_final[k+1, :] =  pi_cmd.dynamics(x_final[k, :], u_list[:,k])

    plt.plot(x_final[:,0], x_final[:,1])

    plt.show()


if __name__ == '__main__':
    main()