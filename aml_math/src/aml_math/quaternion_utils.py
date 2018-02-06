import quaternion
import numpy as np

#quaternion utilities
def quat_convert(q):
    if isinstance(q, np.quaternion):
        return quaternion.as_float_array(q)
    else:
        return q

def quat_conj(q):
    return np.hstack([q[0],-q[1:4]])

def quat_mult(q1, q2):
    s1 = q1[0]
    s2 = q2[0]
    v1 = q1[1:4]
    v2 = q2[1:4]
    return np.hstack([s1*s2 - np.dot(v1,v2), s1*v2 + s2*v1 + np.cross(v1,v2) ])


def compute_log(q):
    v = q[1:4]
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return np.zeros(3)
    else:
        return np.arccos(q[0])*v/norm_v

def compute_exp(w):
    norm_w = np.linalg.norm(w)
    if norm_w == 0:
        return np.zeros(4)
    else:
        return np.hstack([np.cos(norm_w), np.sin(norm_w)*w/norm_w])


def compute_w(q,qdot):
    Q = np.array([[-q[1],-q[2],-q[3]],
                  [q[0],-q[3],q[2]],
                  [q[3],q[0],-q[1]],
                  [-q[2],q[1],q[0]]]);
    w = 2*np.dot(Q.T,qdot);
    return w

def quaternion_from_axis_rotation(axis_rotation):
    rotation_angle = np.linalg.norm(axis_rotation)
    quat = np.zeros(4)
    if(rotation_angle < 1e-4):  # avoid division by zero -- also: can use simpler computation in this case, since for small angles sin(x) = x is a good approximation
        quat[1:4] = axis_rotation/2
    else:
        normalized_axis = axis_rotation / rotation_angle
        quat[1:4] = normalized_axis * np.sin(rotation_angle/2)
    quat[0] = np.sqrt(1 - np.linalg.norm(quat[1:4],2)**2)
    return quat

def rotate_vector(vin, q):
#   // return   ( ( q * quaternion(vin) ) * q_conj ) .complex_part
    vout = quat_mult( quat_mult ( q, np.hstack( np.vstack([0, vin]), np.vstack([q[0],- q[1:4]]) ) ) )
    vout = vout[1:4]
    return vout

def express_vector_in_quat_frame(vin, q):
    vout = rotate_vector(vin, np.vstack([q[0],-q[1:4]]))
    return vout

def euler_to_q(euler):
    c1 = np.cos(euler[2] * 0.5)
    c2 = np.cos(euler[1] * 0.5)
    c3 = np.cos(euler[0] * 0.5)
    s1 = np.sin(euler[2] * 0.5)
    s2 = np.sin(euler[1] * 0.5)
    s3 = np.sin(euler[0] * 0.5)

    q = np.zeros(4)

    q[0] = c1*c2*c3 + s1*s2*s3
    q[1] = c1*s2*c3 + s1*c2*s3
    q[2] = s1*c2*c3 - c1*s2*s3
    q[3] = c1*c2*s3 - s1*s2*c3
    return q

#this is the same as above quat_mult
def quat_multiply(lq, rq):
    #// quaternion entries in order: w, x, y, z
    quat    = np.zeros(4)
    quat[0] = lq[0]*rq[0] - lq[3]*rq[3] - lq[1]*rq[1] - lq[2]*rq[2]
    quat[1] = lq[0]*rq[1] - lq[3]*rq[2] + lq[1]*rq[0] + lq[2]*rq[3]
    quat[2] = lq[0]*rq[2] + lq[3]*rq[1] - lq[1]*rq[3] + lq[2]*rq[0]
    quat[3] = lq[0]*rq[3] + lq[3]*rq[0] + lq[1]*rq[2] - lq[2]*rq[1]
    return quat

#dot product of quaternions
def quat_dot(q1, q2):
    s1 = q1[0]
    s2 = q2[0]
    v1 = q1[1:4]
    v2 = q2[1:4]

    return np.dot(v1,v2) + s1*s2

def quat_norm(q):
    return np.linalg.norm(q)

#linear interpolation between two quaternions
def quat_lerp(q1, q2, t):

    q = q1*(1-t) + q2*t

    return q/quat_norm(q)

#spherical interpolation between two quaternions
def quat_slerp(q1, q2, t):

    dot = quat_dot(q1, q2)
    # /*  dot = cos(theta)
    #     if (dot < 0), q1 and q2 are more than 90 degrees apart,
    #     so we can invert one to reduce spinning */
    if dot < 0.:
        dot = -dot
        q3 = -q2
    else:
        q3 = q2
    
    if dot < 0.95:
        angle = np.arccos(dot)
        return (q1*np.sin(angle*(1-t)) + q3*np.sin(angle*t))/np.sin(angle)
    else: #// if the angle is small, use linear interpolation                               
        return quat_lerp(q1,q3,t)

def compute_omg(q1, q2):
    
    return 2.*compute_log( quat_mult( quat_convert(q1), quat_conj( quat_convert(q2) ) ) )     