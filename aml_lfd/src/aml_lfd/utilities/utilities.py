import numpy as np
import quaternion
from aml_io.io_tools import load_data
from os.path import join, dirname, abspath

def load_demo_data(limb_name, demo_idx, debug=False):

    #get the folder that contains the demos
    data_folder_path = '/'.join(dirname(dirname(abspath(__file__))).split('/')[:-2]) + '/data/'

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

            file_path   = data_folder_path + limb_name + '_demo_data' +  ('_sample_%02d.pkl' % demo_idx)
            
            #loads binary file
            demo_data   = load_data(file_path)
        
        except Exception as e:
            print "Data file cannot be loaded, check demo index..."
            raise e

    return demo_data

def get_sampling_rate(limb_name, demo_idx=None, demo_path=None):

    if demo_path is not None:

        demo_data = load_data(demo_path)
    
    else:

        demo_data = load_demo_data(limb_name=limb_name, demo_idx=demo_idx)

    sampling_rate = demo_data[0].get(0,['sampling_rate'])

    if not sampling_rate:

        sampling_rate = 1.

    else:

        sampling_rate = sampling_rate[0]

    return sampling_rate


def get_effort_sequence_of_demo(limb_name, demo_idx):
    
    demo_data   = load_demo_data(limb_name=limb_name, demo_idx=demo_idx)

    jnt_effort_sequence = []

    for arm_data in demo_data['state']:
        
        jnt_effort_sequence.append(arm_data['effort']-arm_data['gravity_comp'])
  
    return np.asarray(jnt_effort_sequence).squeeze()


def js_inverse_dynamics(limb_name, demo_idx, h_component=True):

    demo_data   = load_demo_data(limb_name=limb_name, demo_idx=demo_idx)
    js_pos_traj, js_vel_traj, js_acc_traj = get_js_traj(limb_name=limb_name, demo_idx=demo_idx)
    tau = np.zeros_like(js_acc_traj)
    
    for k in range(len(js_pos_traj)):
    
        if h_component:

            h = demo_data['state'][k]['gravity_comp']

        else:

            h = np.zeros_like(tau[0])

        tau[k] = np.dot(demo_data['state'][k]['inertia'], js_acc_traj[k]) + h

    return tau

def get_os_traj(limb_name, demo_idx, debug=False):

    #NOT COMPATABLE WITH THE NEW DATA RECORDER, CHANGE THIS FUNCTION
    
    demo_data   = load_demo_data(limb_name=limb_name, demo_idx=demo_idx, debug=debug)
    #the limb on which demo was taken
    ee_pos_traj = []
    ee_vel_traj = []
    ee_acc_traj = []
    ee_ori_traj = []
    ee_omg_traj = []
    ee_ang_traj = []


    if debug:

        ee_pos_list = np.asarray(demo_data).squeeze()

    else:

        for arm_data in demo_data['state']:
            ee_pos_traj.append(arm_data['ee_pos'])
            ee_ori_traj.append(arm_data['ee_ori'])
            
            if arm_data.has_key('ee_vel'):
                ee_vel_traj.append(arm_data['ee_vel'])
            
            if arm_data.has_key('ee_omg'):
                ee_omg_traj.append(arm_data['ee_omg'])

        #check is the list is empty
        if ee_vel_traj:
        
            ee_vel_traj =  np.asarray(ee_vel_traj).squeeze()
            ee_acc_traj =  np.diff(ee_vel_traj, axis=0)
            ee_acc_traj =  np.vstack([np.zeros_like(ee_acc_traj[0]), ee_acc_traj])

        #check is the list is empty
        if  ee_omg_traj:

            ee_omg_traj =  np.asarray(ee_omg_traj).squeeze()
            ee_ang_traj =  np.diff(ee_omg_traj, axis=0)
            ee_ang_traj =  np.vstack([np.zeros_like(ee_ang_traj[0]), ee_ang_traj])

        ee_pos_traj =  np.asarray(ee_pos_traj).squeeze()
        ee_ori_traj =  quaternion.as_quat_array(np.asarray(ee_ori_traj).squeeze())

    return ee_pos_traj, ee_ori_traj, ee_vel_traj, ee_omg_traj, ee_acc_traj, ee_ang_traj

def get_js_traj(limb_name, demo_idx=None, demo_path=None):
    #this function takes the position and velocity of a demonstrated data

    if demo_path is not None:
        demo_data = load_data(demo_path)
    else:
        demo_data   = load_demo_data(limb_name=limb_name, demo_idx=demo_idx, debug=False)
    
    sampling_rate = get_sampling_rate(limb_name=limb_name, demo_path=demo_path)
    
    js_pos_traj = []
    js_vel_traj = []
    
    for k in range(demo_data[0].size):
        
        js_pos_traj.append(demo_data[0].get(k,['position']))
        js_vel_traj.append(demo_data[0].get(k,['velocity']))

    js_pos_traj =  np.asarray(js_pos_traj).squeeze()
    js_vel_traj =  np.asarray(js_vel_traj).squeeze()
    #acceleration trajectory using finite differences
    js_acc_traj =  np.diff(js_vel_traj, axis=0)
    js_acc_traj =  np.vstack([np.zeros_like(js_acc_traj[0]), js_acc_traj])*sampling_rate

    return js_pos_traj, js_vel_traj, js_acc_traj


def plot_demo_data(limb_name, demo_idx):

    ee_list, _, _, _, _, _  = get_os_traj(limb_name=limb_name, demo_idx=demo_idx)

    js_pos_traj, js_vel_traj, js_acc_traj = get_js_traj(limb_name=limb_name, demo_idx=demo_idx)

    #ee_vel_list = np.diff(ee_list, axis=0)
    #print ee_list
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure()
    plt.title('Task space trajectories')
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ee_list[0][0], ee_list[0][1], ee_list[0][2],  linewidths=20, color='r', marker='*')
    ax.scatter(ee_list[-1][0],ee_list[-1][1],ee_list[-1][2], linewidths=20, color='g', marker='*')
    ax.plot(ee_list[:,0],ee_list[:,1],ee_list[:,2])
    #ax.plot(ee_vel_list[:,0],ee_vel_list[:,1],ee_vel_list[:,2], color='m')

    plt.figure(2)
    plt.title('Joint space trajectories')

    num_jnts = js_pos_traj.shape[1]
    plot_no = num_jnts*100 + 11 #just for getting number of joints

    for i in range(num_jnts):
        plt.subplot(plot_no)
        plt.plot(js_pos_traj[:,i])
        plt.plot(js_vel_traj[:,i])
        plt.plot(js_acc_traj[:,i])
        plot_no += 1

    plt.show()

#quaternion utilities

def quat_convert(q):
    if isinstance(q, np.quaternion):
        return quaternion.as_float_array(q)[0]
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