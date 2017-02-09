#!/usr/bin/env python

import numpy as np
import quaternion
import rospy
from aml_robot.baxter_robot import BaxterArm
import baxter_interface
from baxter_interface import CHECK_VERSION
from sensor_msgs.msg import Image
from geometry_msgs.msg  import Pose, PoseStamped
import cv2
from cv_bridge import CvBridge, CvBridgeError
import time
import tf
from tf import TransformListener
from os.path import dirname, abspath

import sys
sys.argv

class BaxterEyeHandCalib():
	
    def __init__(self):
        self.left_arm 	= BaxterArm('left') #object of type Baxter from baxter_mechanism
        self.right_arm  = BaxterArm('right')
        self.aruco_pose  = rospy.Subscriber("/aruco_simple/pose",  Pose, self.aruco_pose_callback)
        self.aruco_pose2 = rospy.Subscriber("/aruco_simple/pose2", Pose, self.aruco_pose2_callback)
        self.aruco_pose3 = rospy.Subscriber("/aruco_single/pose", PoseStamped, self.aruco_pose3_callback)

        self._jnt_calib_positions = None
        self._total_joint_samples = 30
        self._curr_indx  = None
        #to make sure that we get only readings when flags are enabled
        self._capture_img = False
        self._take_pose_reading = False
        self._take_pose2_reading = False
        self.pose_data_filled = False
        self.pose2_data_filled = False
        self.pose_pos = None
        self.pose_ori = None
        self.pose2_pos = None
        self.pose2_ori = None
        self.tf = TransformListener()
        baxter = baxter_interface.RobotEnable(CHECK_VERSION)

        self._calibration_successful = False

        self.br = tf.TransformBroadcaster()
        baxter.enable()

    def broadcast_frame(self, pt, rot, frame_name="marker"):

        rot = np.append(rot,[[0,0,0]],0)
        rot = np.append(rot,[[0],[0],[0],[1]],1)
        quat = tuple(tf.transformations.quaternion_from_matrix(rot))
        now = rospy.Time.now()
        self.br.sendTransform((pt[0], pt[1], pt[2]), tf.transformations.quaternion_from_matrix(rot), now , frame_name, 'base')
        print("should have done it!")

    def aruco_pose_callback(self, data):
        if self._take_pose_reading:
            pos = np.array([data.position.x,data.position.y,data.position.z])
            ori = np.quaternion(data.orientation.w, data.orientation.x, data.orientation.y, data.orientation.z)
            self._take_pose_reading = False
            self.pose_pos = pos
            self.pose_ori = ori
            self.pose_data_filled = True
        else:
            return
    
    def aruco_pose2_callback(self, data):
        #if self._take_pose2_reading:
        pos = np.array([data.position.x,data.position.y,data.position.z])
        ori = np.quaternion(data.orientation.w, data.orientation.x, data.orientation.y, data.orientation.z)
        self._take_pose2_reading = False
        self.pose2_pos = pos
        self.pose2_ori = ori
            #self.pose2_data_filled = True
        #else:
        #    return

    def aruco_pose3_callback(self, data):
        #if self._take_pose2_reading:
        pos = np.array([data.pose.position.x,data.pose.position.y,data.pose.position.z])
        ori = np.quaternion(data.pose.orientation.w, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z)
        self._take_pose2_reading = False
        self.pose2_pos = pos
        self.pose2_ori = ori
            #self.pose2_data_filled = True
        #else:
        #    return
        
    def save_calib_data(self):
        calib_data = {}
        calib_data['left_arm_calib_angle'] = self.left_arm.angles()
        calib_data['right_arm_calib_angle'] = self.right_arm.angles()
        got_common_time = False
        max_attempts = 100
        self._calibration_successful = False
        counter = 0
        while got_common_time is False and counter < max_attempts:
            #some times the tf is busy that it fails to read the time and causes exception
            # this simple hack waits till it reads it and saves it!!
            try:
                counter += 1
                time = self.tf.getLatestCommonTime('openni_rgb_camera', 'base')
                got_common_time = True
            except tf.Exception:
                print("Failed to find common time between openni_rgb_camera and base. Will try again. Attempt count %d/%d"%(counter,max_attempts))
                rospy.sleep(0.1)
            
        if got_common_time:
            translation, rot = self.tf.lookupTransform('openni_rgb_camera', 'base', time)
            calib_data['openni_rgb_camera_pos'] = translation
            calib_data['openni_rgb_camera_ori'] = rot
            np.save('calib_data.npy', calib_data)
            self._calibration_successful = True
        else:
            print("Failed to find transform from openni_rgb_camera to base")

        return calib_data, self._calibration_successful

    def load_calib_data(self):
        # Load
        calib_data_folder_path = dirname(dirname(abspath(__file__))) + '/scripts/'
        try:
            calib_data = np.load(calib_data_folder_path+'calib_data.npy').item()
        except Exception as e:
            print "Caliberation file cannot be loaded"
            raise e
        
        print(calib_data['left_arm_calib_angle']) # displays "world"
        print(calib_data['right_arm_calib_angle'])
        # print(calib_data['openni_rgb_camera_pos'])
        # print(calib_data['openni_rgb_camera_ori'])
        return calib_data

    def self_calibrate(self):
        calib_data = self.load_calib_data()
        self.left_arm.move_to_joint_position(calib_data['left_arm_calib_angle'])
        self.right_arm.move_to_joint_position(calib_data['right_arm_calib_angle'])
        calib_data, calib_success = self.save_calib_data()

        if calib_success:
            print "the openni_rgb_params \n"
            print "postion of the openni camera in robot base frame \t", np.around(calib_data['openni_rgb_camera_pos'],3)
            print "orientation of the openni camera in robot base frame \t", np.around(calib_data['openni_rgb_camera_ori'],3)
        else:
            print "Calibration has failed!"

    def get_box_transform(self):
        flag = True
        while not rospy.is_shutdown():
            #some times the tf is busy that it fails to read the time and causes exception
            # this simple hack waits till it reads it and saves it!!
            try:
                time = self.tf.getLatestCommonTime('box', 'base')
                flag = False
            except tf.Exception:
                pass
            
            if not flag:
                translation, rot = self.tf.lookupTransform('box', 'base', time)
                #self.broadcast_frame(translation, rot, frame_name="marker_box")
                print "translation of box in robot base frame \t", translation
                print "orientation of box in robot base frame \t", rot

                flag = True
                break

def main():
    
    calib = BaxterEyeHandCalib()

    calib.self_calibrate()

    calib.left_arm._ready = False
    calib.right_arm._ready = False
    # calib.get_box_transform()

    sys.exit(0)

if __name__ == '__main__':
    rospy.init_node('baxter_eye_hand_calib_ros_node')
    #get the arguments passed to the script
    cmdargs = str(sys.argv)
    if 'save' in cmdargs:
        calib = BaxterEyeHandCalib()
        calib.save_calib_data()
    
    main()