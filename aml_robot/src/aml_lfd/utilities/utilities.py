import numpy as np
import quaternion
from os.path import dirname, abspath

def load_demo_data(demo_idx=1, debug=False):

    #get the parent folder
    data_folder_path = dirname(dirname(abspath(__file__))) + '/data/'

    if debug:
        #for debugging purposes, load a known trajectory
        try:
            file_path = data_folder_path + 'debug.txt'
            demo_data = np.loadtxt(file_path, dtype='float', delimiter=',')

        except Exception as e:
            print "Data file cannot be loaded"
            raise e

    else:

        try:

            #add the file name of the stored demo file
            file_path   = data_folder_path + 'demo_data_' + str(demo_idx) + '.npy'
            #loads binary file
            demo_data   = np.load(file_path)
        
        except Exception as e:
            print "Data file cannot be loaded, check demo index..."
            raise e

    return demo_data

def get_ee_traj(demo_idx=1, debug=False):
    
    demo_data = load_demo_data(demo_idx=demo_idx, debug=debug)
    ee_pos_list = []
    ee_ori_list = []

    if debug:

        ee_pos_list = np.asarray(demo_data).squeeze()

    else:

        for arm_data in demo_data:
            ee_pos_list.append(arm_data['ee_pos'])
            ee_ori_list.append(arm_data['ee_ori'])

        ee_pos_list =  np.asarray(ee_pos_list).squeeze()
        ee_ori_list =  quaternion.as_quat_array(np.asarray(ee_ori_list).squeeze())

    return ee_pos_list, ee_ori_list

def get_sampling_rate(demo_idx=1):
    demo_data = load_demo_data(demo_idx=demo_idx)
    return demo_data[0]['sampling_rate']

def plot_demo_data(demo_idx=1):

    ee_list, _ = get_ee_traj(demo_idx=demo_idx)
    #ee_vel_list = np.diff(ee_list, axis=0)
    #print ee_list
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ee_list[0][0], ee_list[0][1], ee_list[0][2],  linewidths=20, color='r', marker='*')
    ax.scatter(ee_list[-1][0],ee_list[-1][1],ee_list[-1][2], linewidths=20, color='g', marker='*')
    ax.plot(ee_list[:,0],ee_list[:,1],ee_list[:,2])
    #ax.plot(ee_vel_list[:,0],ee_vel_list[:,1],ee_vel_list[:,2], color='m')
    plt.show()


#quaternion utilities

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

def quat_multiply(lq, rq):
    #// quaternion entries in order: w, x, y, z
    quat    = np.zeros(4)
    quat[0] = lq[3]*rq[3] - lq[0]*rq[0] - lq[1]*rq[1] - lq[2]*rq[2]
    quat[1] = lq[3]*rq[1] - lq[0]*rq[2] + lq[1]*rq[3] + lq[2]*rq[0]
    quat[2] = lq[3]*rq[2] + lq[0]*rq[1] - lq[1]*rq[0] + lq[2]*rq[3]
    quat[3] = lq[3]*rq[0] + lq[0]*rq[3] + lq[1]*rq[2] - lq[2]*rq[1]
    return quat

