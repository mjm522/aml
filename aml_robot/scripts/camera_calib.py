import numpy as np
import quaternion
import rospy
from baxter_robot import BaxterArm
import baxter_interface
from baxter_interface import CHECK_VERSION
from sensor_msgs.msg import Image
from geometry_msgs.msg  import Pose, PoseStamped
import cv2
from cv_bridge import CvBridge, CvBridgeError
import time
import tf
from tf import TransformListener

class Baxter_Eye_Hand_Calib():
	
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
        flag = True
        while flag:
            #some times the tf is busy that it fails to read the time and causes exception
            # this simple hack waits till it reads it and saves it!!
            try:
                time = self.tf.getLatestCommonTime('openni_rgb_camera', 'base')
                flag = False
            except tf.Exception:
                pass
            
            if not flag:
                translation, rot = self.tf.lookupTransform('openni_rgb_camera', 'base', time)
                break

        calib_data['openni_rgb_camera_pos'] = translation
        calib_data['openni_rgb_camera_ori'] = rot
        np.save('calib_data.npy', calib_data)
        return calib_data

    def load_calib_data(self):
        # Load
        try:
            calib_data = np.load('calib_data.npy').item()
        except Exception as e:
            print "Caliberation file cannot be loaded"
            raise e
        
        # print(calib_data['left_arm_calib_angle']) # displays "world"
        # print(calib_data['right_arm_calib_angle'])
        # print(calib_data['openni_rgb_camera_pos'])
        # print(calib_data['openni_rgb_camera_ori'])
        return calib_data

    def self_caliberate(self):
        calib_data = self.load_calib_data()
        self.left_arm.move_to_joint_position(calib_data['left_arm_calib_angle'])
        self.right_arm.move_to_joint_position(calib_data['right_arm_calib_angle'])
        calib_data = self.save_calib_data()
        print "the openni_rgb_params \n"
        print "postion \t", np.around(calib_data['openni_rgb_camera_pos'],3)
        print "orientation \t", np.around(calib_data['openni_rgb_camera_ori'],3)

    def get_box_transform(self):
        flag = True
        while True:
            #some times the tf is busy that it fails to read the time and causes exception
            # this simple hack waits till it reads it and saves it!!
            try:
                time = self.tf.getLatestCommonTime('box', 'base')
                flag = False
            except tf.Exception:
                pass
            
            if not flag:
                translation, rot = self.tf.lookupTransform('box', 'base', time)
                self.broadcast_frame(translation, rot, frame_name="marker_box")
                flag = True

def main():
    
    calib = Baxter_Eye_Hand_Calib()

    calib.self_caliberate()

    #calib.get_box_transform()

if __name__ == '__main__':
    rospy.init_node('baxter_eye_hand_calib_ros_node')
    main()