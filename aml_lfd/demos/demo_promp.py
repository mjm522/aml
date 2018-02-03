import numpy as np
import matplotlib.pyplot as plt
from aml_io.io_tools import load_data
from aml_lfd.promp.promp_ctrl import PROMPCtrl
from aml_lfd.promp.discrete_promp import DiscretePROMP


np.random.seed(0)

#load the data
#this data is generated for a 1D system
#state dimention : 2
#control dimension : 1
data = load_data('data/data_demo_promp.pkl')

demos_list    = [data['states'][k][:,0] for k in range(100)]
Ddemos_list   = [data['states'][k][:,1] for k in range(100)]

#create a promb object by passing the data
d_promp = DiscretePROMP(data=demos_list)
d_promp.train()

def plot_mean_and_sigma(mean, lower_bound, upper_bound, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), lower_bound, upper_bound, color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean)


def demo_generate_traj():

    #add a via point
    # d_promp.add_viapoint(0.7, 5)
    # plt.scatter(0.7, 5, marker='*', s=100)

    #set the start and goal, the spatial scaling
    d_promp.set_start(demos_list[0][0])
    d_promp.set_goal(demos_list[0][-1])

    #add a via point
    # d_promp.add_viapoint(0.3, 2.25)
    # d_promp.add_viapoint(0.6, 2.25)
    # plt.scatter(0.7, 5, marker='*', s=100)

    for traj, traj_vel in zip(demos_list, Ddemos_list):

        plt.figure("ProMP-Pos")
        plt.plot(traj, 'k', alpha=0.2, label='demo_pos')

        plt.figure("ProMP-Vel")
        plt.plot(traj_vel, 'k', alpha=0.2, label='demo_vel')

    for _ in  range(1):

        pos_1, vel_1, acc_1 = d_promp.generate_trajectory(phase_speed=0.8,  randomness=1e-1)
        pos_2, vel_2, acc_2 = d_promp.generate_trajectory(phase_speed=1.,   randomness=1e-1)
        pos_3, vel_3, acc_3 = d_promp.generate_trajectory(phase_speed=1.33, randomness=1e-1)

        plt.figure("ProMP-Pos")
        plt.plot(pos_1, 'r', label='speed=0.8')
        plt.plot(pos_2, 'g', label='speed=1.')
        plt.plot(pos_3, 'b', label='speed=1.33')


        plt.figure("ProMP-Vel")
        plt.plot(vel_1, 'r', label='speed=0.8')
        plt.plot(vel_2, 'g', label='speed=1.')
        plt.plot(vel_3, 'b', label='speed=1.33')

    plt.figure("ProMP-Pos")
    ########################to remove duplicate label handles########################
    handles, labels = plt.gca().get_legend_handles_labels()
    i =1
    while i<len(labels):
        if labels[i] in labels[:i]:
            del(labels[i])
            del(handles[i])
        else:
            i +=1

    plt.legend(handles, labels)
    ###############################################################################

    plt.figure("ProMP-Vel")
    ########################to remove duplicate label handles########################
    handles, labels = plt.gca().get_legend_handles_labels()
    i =1
    while i<len(labels):
        if labels[i] in labels[:i]:
            del(labels[i])
            del(handles[i])
        else:
            i +=1

    plt.legend(handles, labels)
    ###############################################################################


def create_demo_traj():
    """
    This funciton shows how to compute 
    closed form control distribution from the trajectory distribution
    """

    state  = data['states'][0]
    action = data['actions'][0]

    action_dim = 1

    promp_ctl = PROMPCtrl(promp_obj=d_promp)
    promp_ctl.update_system_matrices(A=np.array([ [0.,1.], [0., 0.] ]), B=np.array([ [0.], [1.] ]))

    ctrl_cmds_mean, ctrl_cmds_sigma = promp_ctl.compute_ctrl_traj(state_list=state)

    plt.figure("Ctrl cmds")

    for k in range(action_dim):
        
        mean        = ctrl_cmds_mean[:, k]
        lower_bound = mean - 3.*ctrl_cmds_sigma[:, k, k]
        upper_bound = mean + 3*ctrl_cmds_sigma[:, k, k]

        plot_mean_and_sigma(mean=mean, lower_bound=lower_bound, upper_bound=upper_bound, color_mean='g', color_shading='g')

    plt.plot(action, 'r', label='original control')

    ########################to remove duplicate label handles########################
    handles, labels = plt.gca().get_legend_handles_labels()
    i =1
    while i<len(labels):
        if labels[i] in labels[:i]:
            del(labels[i])
            del(handles[i])
        else:
            i +=1

    plt.legend(handles, labels)
    ###############################################################################



def main():

    demo_generate_traj()
    create_demo_traj()
    plt.show()


if __name__ == '__main__':
    main() 


