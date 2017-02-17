import numpy as np
import quaternion

def convert_rospy_time2sec(rospy_time):
    
    return rospy_time.secs + rospy_time.nsecs*1e-9


def convert_pose_to_euler_tranform(pose):
    '''
    expects a pose vector that is 7 dimensional in the following format
    pose = np.array([x,y,z, quat_x, quat_y, quat_z, quat_w])

    '''

    if len(pose) != 7:
        print "Incorrect pose format"
        raise ValueError


    rot = quaternion.as_rotation_matrix(np.quaternion(pose[6], pose[3], pose[4], pose[5]))

    return np.vstack([np.hstack([rot, np.array([[pose[0]], [pose[1]], [pose[2]]])]),np.array([0.,0.,0.,1.])])
