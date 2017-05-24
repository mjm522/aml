from cost_functions import acc2

duration = 0.5
dt  = 0.01

dmp_params = {
            'path':None,
            'start':None,
            'goal':None, # respecify goals for spatial scaling by changing add_to_goals
            'n_bfs':10, # number of basis function per DMP
            'duration':duration,
            'dt':dt,
             } # how fast the trajectory rolls out


pi2_params = {
    'dmp_params': dmp_params,
    'n_dmps':2,
    'n_bfs':dmp_params['n_bfs'],
    'start':[0., 0.],       # (x, y) the start position of the DMP
    'goal':[3, 1.5],       # (x, y)  the goal position of the DMP
    'duration':0.5,         # the duration of the DMP
    'std':20,               # standard deviation of added noise for parameters
    'n_reps':10,       # repetitions of movement before learning update
    'cost':acc2,            # name of the cost function to use
    'updates':100,          # number of PI2 updates to perform
    'bases_noise':1,        # only add noise to the max active basis function
    'n_reuse':5,            # number of re-used trials per update
    'dt':dt, #integration step size      
}
