#!/usr/bin/env python

import roslib; 
roslib.load_manifest('visualization_marker_tutorials')
roslib.load_manifest('aml_data_collec_utils')

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import rospy
import tf
from tf import TransformListener

from aml_data_collec_utils.config import config

topic = 'visualization_marker_array'
publisher = rospy.Publisher(topic, MarkerArray)

rospy.init_node('various_markers')


box_tf = TransformListener()
flag_box = False

box_length  = config['box_type']['length']
box_breadth = config['box_type']['breadth']
box_height  = config['box_type']['height']

rate = rospy.Rate(30)

while not rospy.is_shutdown():

   markerArray = MarkerArray()

  
   try:
    time = box_tf.getLatestCommonTime('base', 'box')
    flag_box = True
   except tf.Exception:
      print "Some exception occured!!!"

   if flag_box:
      flag_box = False
      translation, ori = box_tf.lookupTransform('base', 'box', time)

      marker_box = Marker()
      marker_box.header.frame_id = "base"
      marker_box.type = marker_box.CUBE  
      marker_box.scale.x = box_length
      marker_box.scale.y = box_height
      marker_box.scale.z = box_breadth
      marker_box.color.a = 1.0
      marker_box.color.r = 0.5
      marker_box.color.g = 1.0
      marker_box.color.b = 0.5
      marker_box.pose.orientation.w = ori[3]
      marker_box.pose.orientation.x = ori[0]
      marker_box.pose.orientation.y = ori[1]
      marker_box.pose.orientation.z = ori[2]
      marker_box.pose.position.x    =  translation[0]
      marker_box.pose.position.y    =  translation[1]
      marker_box.pose.position.z    =  translation[2] - box_height/2

      markerArray.markers.append(marker_box)



   # Publish the MarkerArray
   publisher.publish(markerArray)


   rate.sleep()