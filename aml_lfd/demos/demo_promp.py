import numpy as np
import matplotlib.pyplot as plt
from aml_lfd.promp.discrete_promp import DiscretePROMP
from aml_io.io_tools import load_data

from aml_lfd.promp.promp_ctrl import PROMPCtrl

np.random.seed(0)

def demo_generate_traj(create_data=False):

    """
    Note: the create data generates a data 
    but something is wrong. It doesn't work like the
    demo_list available in demos/data/
    The system performs better if the PhiT in the generate trajectory
    is made time based instead of phase based. -> why? Something else is wrong?
    """

    plt.figure("DiscretePROMP")

    if create_data:

        # Generate and plot trajectory Data
        x = np.arange(0,1.01,0.01)           # time points for trajectories
        nrTraj=30                            # number of trajectoreis for training
        sigmaNoise=0.02                       # noise on training trajectories
        A = np.array([.2, .2, .01, -.05])
        X = np.vstack( (np.sin(5*x), x**2, x, np.ones((1,len(x))) ))
        Y = np.zeros( (nrTraj,len(x)) )

        demos_list = []
        for traj in range(0, nrTraj):
            sample = np.dot(A + sigmaNoise * np.random.randn(1,4), X)[0]
            demos_list.append(sample)

    else:
        """
        generated data, taken from .mat file given
        in promp lib
        """
        demos_list = load_data('./data/data_demo_promp.pkl')

    for traj in demos_list:
        plt.plot(traj, 'k', alpha=0.2)

    #create a promb object by passing the data
    d_promp = DiscretePROMP(data=demos_list)
    d_promp.train()

    #set the start and goal, the spatial scaling
    d_promp.set_start(0.1)
    d_promp.set_goal(0.2)

    #add a via point
    d_promp.add_viapoint(0.3, 2.5)
    d_promp.add_viapoint(0.6, 2.5)


    for _ in  range(1):

        plt.plot(d_promp.generate_trajectory(phase_speed=0.8, randomness=1e-3), 'r')
        plt.plot(d_promp.generate_trajectory(phase_speed=1., randomness=1e-3), 'g')
        plt.plot(d_promp.generate_trajectory(phase_speed=1.03, randomness=1e-3), 'b')

    plt.show()



def demo_ctrl_traj():

    A = np.array([.2, .2, .01, -.05])
    B = np.random.randn(4,2)

    x0 = np.zeros(4)

    demos_list = []
    T = 100
    n_demos = 30
    sigma_noise = 0.01

    u = np.random.randn(2, T)

    for _ in range(n_demos):
        traj = np.zeros([4,T])

        for k in range(T-1):

            traj[:, k+1] = np.dot( (A + sigma_noise * np.random.randn(4)), traj[:, k]) + np.dot(B, u[:, k])

        demos_list.append(traj[0, :].copy())


    #create a promb object by passing the data
    d_promp = DiscretePROMP(data=demos_list)
    d_promp.train()

    c_promp = PROMPCtrl(promp_obj=d_promp, A=A, B=B)

    u_computed = np.zeros(T)

    for t in range(T):
        u_computed[t] = c_promp.compute_gains(t, add_noise=False)


    print "Initial u \n", u
    print "Computed u \n", u_computed



def main():

    # demo_ctrl_traj()
    demo_generate_traj()


if __name__ == '__main__':
    main()


