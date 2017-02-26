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


OS_TORQUE_CNTLR = {
    'rate': 500,
    'kp_p': 0.0001, #gains for position
    'kd_p': np.sqrt(0.),
    'kp_o': 0., #gains for orientation
    'kd_o': np.sqrt(0.), #gains for orientation
    'alpha': 0.0,
    'null_kp': 5.,
    'null_kd': 3.01,
    'use_orientation_ctrl': True,
    'linear_error_thr': 0.12,
    'angular_error_thr': 0.3,
    'js_pos_error_thr' :0.1,
    'deactivate_wait_time': 5,
}


OS_VELCTY_CNTLR = {
    'rate': 500,
    'kp_p': 0.0001, #gains for position
    'kd_p': np.sqrt(0.),
    'kp_o': 0., #gains for orientation
    'kd_o': np.sqrt(0.), #gains for orientation
    'alpha': 0.0,
    'null_kp': 5.,
    'null_kd': 3.01,
    'use_orientation_ctrl': True,
    'linear_error_thr': 0.12,
    'angular_error_thr': 0.3,
    'js_pos_error_thr' :0.1,
    'deactivate_wait_time': 5,
    'integrate_jnt_velocity': True,
}


OS_JT_TORQUE_CNTLR = {
    'rate': 500,
    'kp_p': 0.0001, #gains for position
    'kd_p': np.sqrt(0.),
    'kp_o': 0., #gains for orientation
    'kd_o': np.sqrt(0.), #gains for orientation
    'alpha': 0.0,
    'null_kp': 0.,
    'null_kd': 3.01,
    'use_orientation_ctrl': True,
    'linear_error_thr': 0.12,
    'angular_error_thr': 0.3,
    'js_pos_error_thr' :0.1,
    'pos_threshold': 0.01,
    'deactivate_wait_time': 5,
}


OS_POSTN_CNTLR = {
    'rate': 100,
    'kp_p': 9., #gains for position
    'kd_p': np.sqrt(5.),
    'kp_o': 3.0, #gains for orientation
    'kd_o': np.sqrt(3.0), #gains for orientation
    'alpha': 3.14,
    'null_kp': 4.0,
    'null_kd': 2.5,
    'use_orientation_ctrl': True,
    'pos_threshold': 0.01,
    'linear_error_thr': 0.05, # error of 2 cm
    'angular_error_thr': 0.3, # allowing angular errors of 0.3 radians
    'js_pos_error_thr' :0.1,
    'dt' : 0.01
}

OS_VELCTY_CNTLR = {
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
