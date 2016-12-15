

DISCRETE_DMP = {'gain':1000, # pd gain for trajectory following
        'pen_down':False,
        'threshold':0.1,
        'run_time':1.0,
        'alpha_z':25.,
        'beta_z':4., # respecify goals for spatial scaling by changing add_to_goals
        'bfs':100, # number of basis function per DMP
        'tau':1.0,
        'num_seq':0,
        'num_goals':3,
        'dmps':3,
        'dt':0.01, # how fast the trajectory rolls out
}