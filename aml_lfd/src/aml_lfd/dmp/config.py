
D = 200.

discrete_dmp_config = {
        'end_time':1.0,
        'D':D,
        'K':D**2./4., 
        'rbf_num':500, # number of basis function per DMP
        'tau':1.0,
        'ax':-3,
        'dt':0.001, # how fast the trajectory rolls out
        'original_scaling':1.,
        'type':1,
        'dof':2,
        'goal_thresh':None,
}