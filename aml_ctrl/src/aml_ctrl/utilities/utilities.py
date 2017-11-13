import quaternion
import numpy as np
from aml_ctrl.traj_generator.js_traj_generator import JSTrajGenerator

def quatdiff(quat_curr, quat_des):
    #convert to array is the input arguments are of quaternion type
    if isinstance(quat_des, np.quaternion):
        quat_des = quaternion.as_float_array(quat_des)

    if isinstance(quat_curr, np.quaternion):
        quat_curr = quaternion.as_float_array(quat_curr)

    return quat_des[0]*quat_curr[1:4] - quat_curr[0]*quat_des[1:4] + np.cross(quat_des[1:4],quat_curr[1:4])

def standard_shape_traj(curr_pos, no_set_points=16, shape='circle'):
    
    if shape == 'circle':
        r = 0.21
        th = np.linspace(0., 2*np.pi, no_set_points)
        x = curr_pos[0] + r*np.cos(th)
        y = curr_pos[1] + r*np.sin(th)
        z = np.ones_like(x)*curr_pos[2]
    elif shape == 'eight':
        r  = 0.21
        th =  np.linspace(0., 2*np.pi, no_set_points)
        x = curr_pos[0] + r*np.cos(th)/(1. + np.sin(th)**2)
        y = curr_pos[1] + r*np.cos(th)*np.sin(th)/(1. + np.sin(th)**2)
        z = np.ones_like(x)*curr_pos[2]

    else:
        print "Enter a known shape"
        raise ValueError

    return np.vstack([x,y,z]).T


def pseudo_inv(mat, eps=0.001):
    U, s, V = np.linalg.svd(mat, full_matrices=True)
    tolerance = eps# * np.max(mat, a.rows()) * svd.singularValues().array().abs()(0)
    s[s<tolerance] = tolerance
    return np.dot (V.transpose(), np.dot(np.diag(s**-1.), U.transpose()))


def get_trajectory_from_pkl(file_name, limb_name):
    kwargs = {}
    kwargs['path_to_demo'] = filename
    kwargs['limb_name'] = limb_name
    gen_traj    = JSTrajGenerator(load_from_demo=True, **kwargs)
    trajectory = gen_traj.generate_traj()
    return trajectory
