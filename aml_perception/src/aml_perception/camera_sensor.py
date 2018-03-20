#!/usr/bin/env python

from __future__ import print_function

import roslib
roslib.load_manifest('aml_robot')


import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError

import numpy as np


class CameraSensor(object):

    def __init__(self, image_topic="/camera/rgb/image_rect_color", 
                      depth_topic="/camera/depth_registered/sw_registered/image_rect",
                      cam_info_topic="/camera/rgb/camera_info"):


        # self.rgb_image_pub = rospy.Publisher("rgb_image_out", Image, queue_size=1)
        # self.depth_image_pub = rospy.Publisher("depth_image_out",Image)

        self._bridge = CvBridge()

        self._image_topic = image_topic
        self._depth_topic = depth_topic
        self._camera_info_topic = cam_info_topic

        self._curr_rgb_image = None#np.zeros((480,640),dtype=np.uint8)
        self._curr_depth_image = None#np.zeros((480,640),dtype=np.uint16)
        self._camera_info = None
        self._K = np.eye(3)
        self._received_cam_info = False


        self.start()

    def start(self):
        if self._image_topic is not None:
            self._rgb_image_sub = rospy.Subscriber(self._image_topic, Image, self._on_rgb_image)
        
        if self._depth_topic is not None:
            self._depth_image_sub = rospy.Subscriber(self._depth_topic, Image, self._on_depth_image)

        if self._camera_info_topic is not None:
            self._camera_info_sub = rospy.Subscriber(self._camera_info_topic, CameraInfo, self._on_camera_info)

    def shutdown(self):
        self._rgb_image_sub.shutdown()
        self._depth_image_sub.shutdown()

    def _on_rgb_image(self, data):

        try:
            cv_image = self._bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)

        # (rows,cols,channels) = cv_image.shape
        # if cols > 60 and rows > 60 :
        #   cv2.circle(cv_image, (50,50), 10, 255)

        self._curr_rgb_image = np.array(cv_image, dtype=np.uint8)

        # cv2.imshow("RGB Image window", self._curr_rgb_image)
        # cv2.waitKey(1)

        #try:
        #  image_msg = self._bridge.cv2_to_imgmsg(self._curr_rgb_image, "bgr8")
        #  self.rgb_image_pub.publish(image_msg)
        #except CvBridgeError as e:
        #  print("camera_sensor: cv_bridge error. Unable to publish rgb_image")
        #except Exception as e:
        #  print("camera_sensor: other error. Unable to publish rgb_image")

    def _on_depth_image(self,data):

        try:
            cv_image = self._bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        except CvBridgeError as e:
          print("camera_sensor: unable to convert depth_image to opencv type",e)
          # print 'camera_sensor: unable to convert depth_image to opencv type'
        # except Exception as e:
        #   print "camera_sensor: other error."
        
        self._curr_depth_image = np.array(cv_image,dtype=np.float32)
        self._curr_depth_image[np.isnan(self._curr_depth_image)] = 0


        # try:
        #     # img = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        #     # cv2.imshow("Depth Image window", img)
        #     # print("Max",max_val,"Min",min_val)
        #     # cv_image.convertTo(B,CV_8U,255.0/(Max-Min));
        #     # self.depth_image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        # except CvBridgeError as e:
        #     print(e)


    def _on_camera_info(self,data):

        if not self._received_cam_info:

            self._camera_info = data
            self._width = data.width
            self._height = data.height
            self._received_cam_info = True

    
            self.set_intrinsics(fx = self._camera_info.K[0],
                                fy = self._camera_info.K[4],
                                cx = self._camera_info.K[2],
                                cy = self._camera_info.K[5])

            self._camera_info_sub.unregister()



    def set_intrinsics(self, fx, fy, cx, cy):

        # set camera projection matrix
        self._K = np.array([[fx,0,  cx],
                            [0, fy, cy],
                            [0, 0,  1]])

    def intrinsics(self):
        return self._K

    def rgb_image(self):
        return self._curr_rgb_image

    def depth_image(self):
        return self._curr_depth_image


    def cloud(self):

        return self.deproject(self.depth_image())


    def deproject(self, depth_image):
        '''
        Deprojects a depth image (2D numpy float array) into a point cloud
        Params:

        depth_image: (HxW numpy array of floats) 2D depth image to project
        Returns:
        3xN numpy float array of 3D points
        '''
        height = depth_image.shape[0]
        width = depth_image.shape[1]

        # create homogeneous pixels 
        row_indices = np.arange(height)
        col_indices = np.arange(width)
        pixel_grid = np.meshgrid(col_indices, row_indices)
        pixels = np.c_[pixel_grid[0].flatten(), pixel_grid[1].flatten()].T
        pixels_homog = np.r_[pixels, np.ones([1, pixels.shape[1]])]
        depth_arr = np.tile(depth_image.flatten(), [3,1])

        # deproject
        points_3d = depth_arr * np.linalg.inv(self._K).dot(pixels_homog)

    def project(self, points):
        '''
        Projects a set of points into the camera given by these parameters
                
        Params:
        points: (3xN numpy array of floats) 3D points to project
        Returns:
        2xN numpy float array of 2D image coordinates
        1xN binary numpy array indicating whether or not point projected outside of image
        '''
        # check valid data

        if points.shape[0] != 3:
            raise Exception('Incorrect data dimension. CameraParams project must be supplied a 3xN numpy float array.')

        points_proj = self.K_.dot(points)
        point_depths = np.tile(points_proj[2,:], [3, 1])
        points_proj = np.divide(points_proj, point_depths)
        points_proj = np.round(points_proj)

        # find valid indices
        valid = (points_proj[0,:] >= 0) & (points_proj[1,:] >= 0) & (points_proj[0,:] < self.width_) & (points_proj[1,:] < self.height_)

        return points_proj[:2,:].astype(np.int), np.where(valid)[0]



def main(args):

  ic = CameraSensor(image_topic='/usb_cam/image_raw', depth_topic=None, cam_info_topic='/usb_cam/camera_info')
  rospy.init_node('camera_sensor_node', anonymous=True)

  try:

    while not rospy.is_shutdown():

        cv2.imshow("Image", ic.rgb_image())

        print(ic.intrinsics())

        cv2.waitKey(1)

  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)