from dmp_shell import DiscreteDMPShell
import matplotlib.pyplot as plt

def main():
    start = 0
    goal = 1

    dmp_params = {
            'path':None,
            'start':start,
            'goal':goal, # respecify goals for spatial scaling by changing add_to_goals
            'n_bfs':10, # number of basis function per DMP
            'duration':1.0,
            'dt':0.01,
             } # how fast the trajectory rolls out

    #train dmp
    dmp = DiscreteDMPShell(dmp_params)

    #test dmp
    tau = 4.5 #vary this for temporal scaling
    new_start = 0 #vary this for spatial scaling
    new_goal = 1.0 #vary this for spatial scaling

    dmp.reset_state_dmp(new_start)
    dmp.set_goal(new_goal)

    y_track,dy_track,ddy_track = dmp.rollout_dmp(tau)

    plt.figure(1)
    plt.plot(dmp._true_traj[0,:], lw=2, color='r')
    plt.plot(y_track[:,0], lw=2, color='g')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()