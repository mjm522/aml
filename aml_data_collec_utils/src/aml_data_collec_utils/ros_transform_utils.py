import numpy as np
import rospy
import tf

##### UTILS #######

def get_pose(tf_listener, target, source, time, time_out=4.0):

    try:
        tf_listener.waitForTransform(target, source, time, rospy.Duration(time_out))
        translation, rot = tf_listener.lookupTransform(target, source, time)
    except Exception as e:
        return
    
    transform = pq_to_transform(tf_listener,translation,rot)

    return transform

def transform_to_pq(transform):
    """
    tf: a ros transform listener
    transform: a 4x4 homogenous transformation matrix
    return: a translation p, and a quaternion q
    """

    p = tf.transformations.translation_from_matrix(transform)
    q = tf.transformations.quaternion_from_matrix(transform)
    
    return p, q

def pq_to_transform(tf_listener,p,q):

    # Returns a 4x4 homogeneous transformation matrix composed of [R | t]
    #                                                             [0 | 1]
    # where R is a rotation matrix, and t is a translation

    transform = np.asmatrix(tf_listener.fromTranslationRotation(p, q))

    return transform