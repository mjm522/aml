#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import Pose, PoseStamped
import tf
from tf import TransformListener

from aml_calib.hand_eye_calib import HandEyeCalib

import sys

sys.argv


class HandEyeCalibrator(object):
    def __init__(self):
        self._camera_frame = "camera_link"
        self._world_frame = "marker26"
        self._robot_base_frame = "base"
        self._ee_frame = "right_hand"

        self._calib = HandEyeCalib()

        self._br = tf.TransformBroadcaster()
        self._tf = TransformListener()

    def broadcast_frame(self, pt, rot, frame_name="marker"):

        rot = np.append(rot, [[0, 0, 0]], 0)
        rot = np.append(rot, [[0], [0], [0], [1]], 1)
        quat = tuple(tf.transformations.quaternion_from_matrix(rot))
        now = rospy.Time.now()
        self.br.sendTransform((pt[0], pt[1], pt[2]), tf.transformations.quaternion_from_matrix(rot), now, frame_name,
                              'base')
        print("should have done it!")


    def get_transform(self, base_frame, source_frame):

        if base_frame == source_frame:
            return (0, 0, 0), (0, 0, 0, 1), rospy.Time.now()

        # t = (0,0,0)
        # q = (0,0,0,1)
        time = None

        while time is None:
            try:

                time = self._tf.getLatestCommonTime(source_frame, base_frame)
            except:
                rospy.loginfo("Failed to get common time between %s and %s. Trying again..."%(source_frame,base_frame,))

        t = None
        q = None
        try:
            t, q = self._tf.lookupTransform(base_frame, source_frame, time)#self._tf_buffer.lookup_transform(frame_name, base_frame, rospy.Time(0), rospy.Duration(5.0))

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Transform could not be queried.")

            return (0, 0, 0), (0, 0, 0, 1), rospy.Time.now()

        translation = (t[0], t[1], t[2])
        quaternion = (q[0], q[1], q[2], q[3])
        # translation = (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z)
        # quaternion = (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w)

        return translation, quaternion, time

    #
    # def save_calib_data(self):
    #     calib_data = {}
    #     calib_data['left_arm_calib_angle'] = self.left_arm.angles()
    #     calib_data['right_arm_calib_angle'] = self.right_arm.angles()
    #     got_common_time = False
    #     max_attempts = 100
    #     self._calibration_successful = False
    #     counter = 0
    #     while got_common_time is False and counter < max_attempts:
    #         # some times the tf is busy that it fails to read the time and causes exception
    #         # this simple hack waits till it reads it and saves it!!
    #         try:
    #             counter += 1
    #             time = self.tf.getLatestCommonTime('openni_rgb_camera', 'base')
    #             got_common_time = True
    #         except tf.Exception:
    #             print(
    #                 "Failed to find common time between openni_rgb_camera and base. Will try again. Attempt count %d/%d" % (
    #                     counter, max_attempts))
    #             rospy.sleep(0.1)
    #
    #     if got_common_time:
    #         translation, rot = self.tf.lookupTransform('openni_rgb_camera', 'base', time)
    #         calib_data['openni_rgb_camera_pos'] = translation
    #         calib_data['openni_rgb_camera_ori'] = rot
    #         np.save('calib_data.npy', calib_data)
    #         self._calibration_successful = True
    #     else:
    #         print("Failed to find transform from openni_rgb_camera to base")
    #
    #     return calib_data, self._calibration_successful
    #
    # def load_calib_data(self):
    #     # Load
    #     calib_data_folder_path = dirname(dirname(abspath(__file__))) + '/scripts/'
    #     try:
    #         calib_data = np.load(calib_data_folder_path + 'calib_data.npy').item()
    #     except Exception as e:
    #         print "Caliberation file cannot be loaded"
    #         raise e
    #
    #     print(calib_data['left_arm_calib_angle'])  # displays "world"
    #     print(calib_data['right_arm_calib_angle'])
    #     # print(calib_data['openni_rgb_camera_pos'])
    #     # print(calib_data['openni_rgb_camera_ori'])
    #     return calib_data

    def to_transform_matrix(self, t, q):

        t_mat = tf.transformations.translation_matrix(t)
        r_mat = tf.transformations.quaternion_matrix(q)
        transform_mat = np.dot(t_mat, r_mat)

        return transform_mat


    def calibrate_from_file(self):


        calib_data = np.load('hand_eye_calibration_test2.npy').item()

        camera_poses = calib_data['camera_poses']
        ee_poses = calib_data['ee_poses']

        for i in range(len(camera_poses)):
            self._calib.add_measurement(ee_poses[i], camera_poses[i])

        hand_eye_transform = self._calib.calibrate()

        print 'Hand-eye transform: ', hand_eye_transform


    def calibrate(self):

        calibrate = False

        while not calibrate:

            camera_t, camera_q, _ = self.get_transform(self._world_frame, self._camera_frame)
            ee_t, ee_q, _ = self.get_transform(self._robot_base_frame, self._ee_frame)

            camera_transform = self.to_transform_matrix(camera_t,camera_q)
            ee_transform = self.to_transform_matrix(ee_t, ee_q)


            print "Camera transform: ", camera_t, camera_q
            print "Hand transform: ", ee_t, ee_q
            r = raw_input("Press enter to record to next. Add/Skip/Calibrate? (a/s/c)")

            if r == 'a':
                self._calib.add_measurement(ee_transform, camera_transform)
                print 'Transform has been added for calibration'
            elif r == 'c':
                calibrate = True
            else:
                print "Skipping transformations...\n\n"
        hand_eye_transform = self._calib.calibrate()

        calib_data = {'hand_eye_transform':  hand_eye_transform,
                      'camera_poses': self._calib._camera_poses,
                      'ee_poses': self._calib._ee_poses}

        np.save('hand_eye_calibration.npy', calib_data)







def main():
    calib = HandEyeCalibrator()

    calib.calibrate_from_file()


    sys.exit(0)


if __name__ == '__main__':
    rospy.init_node('aml_hand_eye_calibrator')

    main()
