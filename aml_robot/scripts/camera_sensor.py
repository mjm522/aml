#!/usr/bin/env python

from __future__ import print_function

import roslib
roslib.load_manifest('manipulation')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import numpy as np



class CameraSensor(object):

  def __init__(self):
    self.rgb_image_pub = rospy.Publisher("rgb_image_out",Image)
    self.depth_image_pub = rospy.Publisher("depth_image_out",Image)

    self.bridge = CvBridge()
    self.rgb_image_sub = rospy.Subscriber("/camera/rgb/image_rect_color",Image,self._on_rgb_image)
    self.depth_image_sub = rospy.Subscriber("/camera/depth_registered/sw_registered/image_rect",Image,self._on_depth_image)

    self.curr_rgb_image = np.zeros((480,640),dtype=np.uint8)
    self.curr_depth_image = np.zeros((480,640),dtype=np.uint16)

  def _on_rgb_image(self,data):
    try:
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    (rows,cols,channels) = cv_image.shape
    if cols > 60 and rows > 60 :
      cv2.circle(cv_image, (50,50), 10, 255)

    color_array = np.array(cv_image,dtype=np.uint8)
    self.curr_rgb_image = color_array
    # print type(cv_image), cv_image.dtype

    cv2.imshow("RGB Image window", cv_image)
    cv2.waitKey(3)

    

    try:
      self.rgb_image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)

  def _on_depth_image(self,data):
    try:
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    except CvBridgeError as e:
        print(e)
    
    depth_array = np.array(cv_image,dtype=np.float32)
    depth_array[np.isnan(depth_array)] = 0



    # depth_array = np.array(cv_image.data,dtype=np.float32)/255.0

    # cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
    # print type(cv_image), cv_image.dtype, cv_image.shape
    


    # print(depth_array.shape)    
    # # cv2.imshow("Depth Image window", cv_image)
    # print(np.max(depth_array))
    cv2.waitKey(3)

    try:
        
        # tmp = None
        # depth=np.array(cv_image.data,dtype=np.float32)
        # print(depth)
        
        # max_val = np.max(depth_array)
        # min_val = np.min(depth_array)

        # print(len(cv_image.data))
        # vis2 = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Depth Image window", img)
        # print("Max",max_val,"Min",min_val)
        # cv_image.convertTo(B,CV_8U,255.0/(Max-Min));
        # self.depth_image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
    except CvBridgeError as e:
        print(e)
        print("error!")

def main(args):

  ic = CameraSensor()
  rospy.init_node('camera_sensor_node', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)