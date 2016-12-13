import numpy as np


CONTROLLER = {
    'rate': 100,
    'kp': 10.,
    'kd': np.sqrt(10.),
    'alpha': 3.25,
    'null_kp':  1.5*10.,
    'null_kd': 2.5*10.,
}


OSC_TORQUE_CNTLR = {
    'rate': 100,
    'kp': 10.,
    'kd': np.sqrt(10.),
    'alpha': 3.25,
    'null_kp':  1.5*10.,
    'null_kd': 2.5*10.,
}

OSC_POSTN_CNTLR = {
    'rate': 100,
    'kp': 10.,
    'kd': np.sqrt(10.),
    'alpha':3.25,
    'null_kp': 1.5*10.,
    'null_kd': 2.5*10.,
    'pos_threshold':0.01,
}

OSC_VELCTY_CNTLR = {
    'rate': 100,
    'kp': 10.,
    'kd': np.sqrt(10.),
    'alpha': 3.25,
    'null_kp': 1.5*10.,
    'null_kd': 2.5*10.,
}