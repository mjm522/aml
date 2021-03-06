import numpy as np

CONTROLLER = {
    'rate': 100,
    'kp': 10.,
    'kd': np.sqrt(10.),
    'velocity_low_pass_alpha': 0.01,
    'js_pos_error_thr' :0.1,
    'deactivate_wait_time': 5,
}

JS_POSTN_CNTLR = {
    'rate': 500,
    'kp_q': 0.1, #gains for position
    'kd_dq': np.sqrt(0.0001),
    'velocity_low_pass_alpha': 0.01,
    'js_pos_error_thr' :0.1,
    'deactivate_wait_time': 5,
}


JS_VELOCITY_CNTLR = {
    'rate': 500,
    'kp_p': 0.1, #gains for position
    'kd_p': np.sqrt(0.1),
    'alpha': 0.0,
    'null_kp': 5.,
    'null_kd': 3.01,
    'js_pos_error_thr' :0.1,
    'deactivate_wait_time': 5,
    'integrate_jnt_velocity': True,
    'velocity_filter_alpha': 0.05 # low pass filter on the robot velocity readings
}


JS_TORQUE_CNTLR = {
    'rate': 500,
    'kp_q': 5., #gains for position
    'kd_dq': np.sqrt(10.),
    'js_pos_error_thr' :0.1,
    'deactivate_wait_time': 5,
}

JS_POSTN_CNTLR_BAXTER = {
    'rate': 500,
    'kp_q': 0.1, #gains for position
    'kd_dq': np.sqrt(0.0001),
    'velocity_low_pass_alpha': 0.01, # low pass filter on the robot velocity readings
    'js_pos_error_thr' :0.1,
    'deactivate_wait_time': 5,
}

JS_VELOCITY_CNTLR_BAXTER = {
    'rate': 500,
    'kp_q': 5.0, #gains for position
    'kd_dq': np.sqrt(0.01),
    'velocity_low_pass_alpha': 0.01, # low pass filter on the robot velocity readings
    'js_pos_error_thr' :0.1,
    'deactivate_wait_time': 5,
}

JS_TORQUE_CNTLR_BAXTER = {
    'rate': 500,
    'kp_q': 15.0, #gains for position
    'kd_dq': np.sqrt(5.),
    'velocity_low_pass_alpha': 0.01, # low pass filter on the robot velocity readings
    'js_pos_error_thr' :0.05,
    'deactivate_wait_time': 5,
}


JS_POSTN_CNTLR_SAWYER = {
    'rate': 500,
    'kp_q': 10., #gains for position
    'kd_dq': np.sqrt(10.),
    'velocity_low_pass_alpha': 0.01, # low pass filter on the robot velocity readings
    'js_pos_error_thr' :0.09,
    'deactivate_wait_time': 5,
}

JS_POSTN_CNTLR_SAWYER_PISA_BULLET = {
    'rate': 500,
    'kp_q': 10., #gains for position
    'kd_dq': np.sqrt(10.),
    'velocity_low_pass_alpha': 0.01, # low pass filter on the robot velocity readings
    'js_pos_error_thr' :0.09,
    'deactivate_wait_time': 5,
}

JS_VELOCITY_CNTLR_SAWYER_PISA_BULLET = {
    'rate': 500,
    'kp_q': 5.0, #gains for position
    'kd_dq': np.sqrt(0.01),
    'velocity_low_pass_alpha': 0.01, # low pass filter on the robot velocity readings
    'js_pos_error_thr' :0.1,
    'deactivate_wait_time': 5,
}


JS_TORQUE_CNTLR_SAWYER_PISA_BULLET = {
    'rate': 500,
    'kp_q': 5.0, #gains for position
    'kd_dq': np.sqrt(5.),
    'velocity_low_pass_alpha': 0.01, # low pass filter on the robot velocity readings
    'js_pos_error_thr' :0.1,
    'deactivate_wait_time': 5,
}


JS_POSTN_CNTLR = JS_POSTN_CNTLR_BAXTER
JS_VELOCITY_CNTLR = JS_VELOCITY_CNTLR_BAXTER
JS_TORQUE_CNTLR = JS_TORQUE_CNTLR_BAXTER

ALL_CONFIGS = {

    'js_position_baxter': JS_POSTN_CNTLR_BAXTER,
    'js_velocity_baxter': JS_VELOCITY_CNTLR_BAXTER,
    'js_torque_baxter': JS_TORQUE_CNTLR_BAXTER,
    'js_position_sawyer_pisa_bullet': JS_POSTN_CNTLR_SAWYER_PISA_BULLET,
    'js_velocity_sawyer_bullet': JS_VELOCITY_CNTLR_SAWYER_PISA_BULLET,
    'js_torque_sawyer_bullet': JS_TORQUE_CNTLR_SAWYER_PISA_BULLET,

}
