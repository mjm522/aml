#!/usr/bin/env python

import roslib; roslib.load_manifest('aml_grasp')
import rospy
import tf
import tf2_ros

import atexit

finger_link = 'soft_hand_thumb_distal_link'

from aml_grasp.glove_experiments.transform_udp_receiver import MocapTransformReceiver
import numpy as np

from aml_io.io_tools import save_data


class GroundTruthExperiment(object):
    def __init__(self):

        rospy.init_node('glove_ground_truth_experiment', anonymous=True)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self._mocap_listener = MocapTransformReceiver()

        self._data_finger = []
        self._data_finger_gd = []

        atexit.register(self.save)

    def get_transform(self, frame_name, base_frame):

        if frame_name == base_frame:
            return (0, 0, 0), (0, 0, 0, 1), rospy.Time.now()

        # t = (0,0,0)
        # q = (0,0,0,1)
        t = None
        try:
            t = self._tf_buffer.lookup_transform(frame_name, base_frame, rospy.Time(0), rospy.Duration(5.0))

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Transform could not be queried.")

            return (0, 0, 0), (0, 0, 0, 1), rospy.Time.now()

        translation = (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z)
        quaternion = (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w)

        return translation, quaternion, t.header.stamp

    def to_transform_matrix(self, t, q):
        t_mat = tf.transformations.translation_matrix(t)
        r_mat = tf.transformations.quaternion_matrix(q)
        transform_mat = np.dot(t_mat, r_mat)

        return transform_mat

    def inverse_transform(self,t, q):

        mat = self.to_transform_matrix(t,q)

        return self.from_transform_matrix(np.linalg.inv(mat))

    def from_transform_matrix(self, transform):

        t = tf.transformations.translation_from_matrix(transform)
        q = tf.transformations.quaternion_from_matrix(transform)

        return t, q

    def multiply_transforms(self, t1, q1, t2, q2):

        transform_mat1 = self.to_transform_matrix(t1, q1)
        transform_mat2 = self.to_transform_matrix(t2, q2)

        transform_mat3 = np.dot(transform_mat1, transform_mat2)

        t = tf.transformations.translation_from_matrix(transform_mat3)
        q = tf.transformations.quaternion_from_matrix(transform_mat3)

        return t, q


    def save(self):
        print "Saving data"
        save_data({'Measurement': self._data_finger, 'GroundTruth': self._data_finger_gd}, "little_experiment1.pkl")

    def run_experiment(self):

        while not rospy.is_shutdown():
            t_finger, q_finger, _ = self.get_transform('base', finger_link)
            t_finger_gd, q_finger_gd = self._mocap_listener.read_data()

            transform_finger = np.hstack([t_finger,q_finger])
            transform_finger_gd = np.hstack([t_finger_gd,q_finger_gd])

            self._data_finger.append(transform_finger)
            self._data_finger_gd.append(transform_finger_gd)

            print "Measurement: ", transform_finger
            print "GroundTruth: ", transform_finger_gd




if __name__ == '__main__':
    try:
        s = GroundTruthExperiment()

        s.run_experiment()
    except rospy.ROSInterruptException:
        pass
