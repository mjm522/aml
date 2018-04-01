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

OS_VELCTY_CNTLR = {
    'rate': 500,
    'kp_p': 5.0, #gains for position
    'kd_p': np.sqrt(0.5),
    'kp_o': 5.0, #gains for orientation
    'kd_o': np.sqrt(0.001), #gains for orientation
    'alpha': 3.14,
    'null_kp': 5.*0,
    'null_kd': 3.01*0,
    'use_orientation_ctrl': True,
    'linear_error_thr': 0.05,
    'angular_error_thr': 0.05,
    'deactivate_wait_time': 5,
}



OS_POSTN_CNTLR_SIM_BAXTER = {
    'rate': 100,
    'kp_p': 5., #gains for position
    'kd_p': np.sqrt(0.0001),
    'kp_o': 8.0, #gains for orientation
    'kd_o': np.sqrt(0.0001), #gains for orientation
    'alpha': 3.14,
    'null_kp': 4.0*0,
    'null_kd': 2.5*0,
    'use_orientation_ctrl': True,
    'linear_error_thr': 0.02, # error of 2 cm
    'angular_error_thr': 0.02, # allowing angular errors of 0.3 radians
    'js_pos_error_thr' :0.1,
    'dt' : 0.01
}

OS_POSTN_CNTLR_REAL_BAXTER = {
    'rate': 500,
    'kp_p': 10., #gains for position
    'kd_p': np.sqrt(10.0),
    'kp_o': 8.0, #gains for orientation
    'kd_o': np.sqrt(8.0), #gains for orientation
    'alpha': 0.0,
    'null_kp': 4.0*0,
    'null_kd': 2.5*0,
    'use_orientation_ctrl': True,
    'linear_error_thr': 0.02, # error of 2 cm
    'angular_error_thr': 0.02, # allowing angular errors of 0.3 radians
    'js_pos_error_thr' :0.1,
    'dt' : 0.01
}

OS_POSTN_CNTLR_REAL_SAWYER = {
    'rate': 200,
    'kp_p': 10.0, #gains for position
    'kd_p': np.sqrt(0.0),
    'kp_o': 8.0, #gains for orientation
    'kd_o': np.sqrt(0.0), #gains for orientation
    'alpha': 0.0,
    'null_kp': 4.0*0,
    'null_kd': 2.5*0,
    'use_orientation_ctrl': True,
    'linear_error_thr': 0.025, # error of 2 cm
    'angular_error_thr': 0.025, # allowing angular errors of 0.3 radians
    'js_pos_error_thr' :0.1,
    'dt' : 0.01
}


OS_TORQUE_CNTLR_SIM_BAXTER = {
    'rate': 500,
    'kp_p': 5.0,
    'kd_p': np.sqrt(0.5),
    'kp_o': 5.0, #gains for orientation
    'kd_o': np.sqrt(0.0001), #gains for orientation
    'alpha': 3.14,
    'null_kp': 1.5,
    'null_kd': 2.5,
    'use_orientation_ctrl': True,
    'linear_error_thr': 0.05,
    'angular_error_thr': 0.05,
    'integrate_jnt_velocity': False,
    'deactivate_wait_time': 10
}

# [1.25*10.0, 0.95*10.0, 0.65*10, 0.65*10, 0.04*10, 0.45*10, 0.0005*]
OS_TORQUE_CNTLR_REAL_BAXTER = {
    'rate': 500,
    'kp_p': 15.0,
    'kd_p': np.sqrt(15.0),
    'kp_o': 10.0, #gains for orientation
    'kd_o': np.sqrt(10.0), #gains for orientation
    'alpha': 3.14,
    'null_kp': 1.5,
    'null_kd': 2.5,
    'use_orientation_ctrl': True,
    'linear_error_thr': 0.025,
    'angular_error_thr': 0.025,
    'integrate_jnt_velocity': False,
    'deactivate_wait_time': 10
}

OS_TORQUE_CNTLR_REAL_SAWYER = {
    'rate': 500,
    'kp_p': 50.0,
    'kd_p': np.sqrt(0.0),
    'kp_o': 8.0, #gains for orientation
    'kd_o': np.sqrt(0.0), #gains for orientation
    'alpha': 3.14*0,
    'null_kp': 1.5*0,
    'null_kd': 2.5*0,
    'use_orientation_ctrl': True,
    'linear_error_thr': 0.025,
    'angular_error_thr': 0.01,
    'integrate_jnt_velocity': False,
    'deactivate_wait_time': 50
}


OS_IMPEDANCE_CNTLR = {
    'rate': 500,
    'kp_p': 0.0001, #gains for position
    'kd_p': np.eye(3)*0.01,
    'kp_o': 0., #gains for orientation
    'kd_o': np.sqrt(0.), #gains for orientation
    'Md':np.eye(3),
    'kd_q':np.eye(7)*0.01,
    'alpha': 0.0,
    'null_kp': 0.,
    'null_kd': 3.01,
    'use_orientation_ctrl': False,
    'linear_error_thr': 0.12,
    'angular_error_thr': 0.3,
    'js_pos_error_thr' :0.1,
    'pos_threshold': 0.01,
    'deactivate_wait_time': 5,
}



ALL_CONFIGS = {
    
    'position_baxter': OS_POSTN_CNTLR_REAL_BAXTER,
    'torque_baxter': OS_TORQUE_CNTLR_REAL_BAXTER,
    'position_sawyer' : OS_POSTN_CNTLR_REAL_SAWYER,
    'torque_sawyer': OS_TORQUE_CNTLR_REAL_SAWYER,

}

OS_POSTN_CNTLR = OS_POSTN_CNTLR_REAL_BAXTER
OS_TORQUE_CNTLR = OS_TORQUE_CNTLR_REAL_BAXTER
