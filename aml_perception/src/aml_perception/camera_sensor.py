#!/usr/bin/env python

from __future__ import print_function

import roslib
roslib.load_manifest('aml_robot')


import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import numpy as np


class CameraSensor(object):

  def __init__(self,image_topic="/camera/rgb/image_rect_color", depth_topic="/camera/depth_registered/sw_registered/image_rect"):
    self.rgb_image_pub = rospy.Publisher("rgb_image_out", Image, queue_size=1)
    # self.depth_image_pub = rospy.Publisher("depth_image_out",Image)

    self._bridge = CvBridge()

    self._image_topic = image_topic
    self._depth_topic = depth_topic
    
    self.start()

    self._curr_rgb_image = None#np.zeros((480,640),dtype=np.uint8)
    self._curr_depth_image = None#np.zeros((480,640),dtype=np.uint16)

  def start(self):
    self._rgb_image_sub = rospy.Subscriber(self._image_topic,Image,   self._on_rgb_image)
    self._depth_image_sub = rospy.Subscriber(self._depth_topic,Image, self._on_depth_image)

  def shutdown(self):
    self._rgb_image_sub.shutdown()
    self._depth_image_sub.shutdown()

  def _on_rgb_image(self, data):
    pass
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
      print("camera_sensor: unable to publish depth_image")
    except Exception as e:
      print("camera_sensor: other error. Unable to publish rgb_image")
    
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