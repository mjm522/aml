import numpy as np


CONTROLLER = {
    'rate': 100,
    'kp': 10.,
    'kd': np.sqrt(10.),
    'alpha': 3.25,
    'null_kp':  1.5*10.,
    'null_kd': 2.5*10.,
    'use_orientation_ctrl': True,
    'linear_error_thr': 0.12,
    'angular_error_thr': 0.1
}


OSC_TORQUE_CNTLR = {
    'rate': 500,
    'kp': 10.,
    'kd': np.sqrt(10.),
    'alpha': 3.25,
    'null_kp':  1.0*10.,
    'null_kd': 0.25*10.,
    'use_orientation_ctrl': True,
    'linear_error_thr': 0.12,
    'angular_error_thr': 0.3,
    'deactivate_wait_time': 5,
}

OSC_POSTN_CNTLR = {
    'rate': 100,
    'kp': 20.0,
    'kd': np.sqrt(10.),
    'alpha': 3.14,
    'null_kp': 1.0*10.,
    'null_kd': 1.0*10.,
    'use_orientation_ctrl': True,
    'pos_threshold': 0.0001,
    'linear_error_thr': 0.05, # error of 2 cm
    'angular_error_thr': 0.3, # allowing angular errors of 0.3 radians
    'dt' : 0.01
}

OSC_VELCTY_CNTLR = {
    'rate': 100,
    'kp': 10.,
    'kd': np.sqrt(10.),
    'alpha': 3.25,
    'null_kp': 1.5*10.,
    'null_kd': 2.5*10.,
    'use_orientation_ctrl': True,
    'linear_error_thr': 0.12,
    'angular_error_thr': 0.1
}