#!/usr/bin/env python

import roslib; 
roslib.load_manifest('visualization_marker_tutorials')
roslib.load_manifest('aml_data_collec_utils')

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import rospy
import tf
from tf import TransformListener

import numpy as np

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

next_id = 10

prev_trans = None

def make_box_marker(trans, ori, id = None, r = 0.5, g = 1.0, b = 0.5, a = 1.0, ns = '', lifetime = rospy.Duration(0)):

   global next_id, prev_trans

   marker_box = Marker()

   dist = 0.0
   if id is None and prev_trans is not None:
      dist = np.linalg.norm(np.array(prev_trans)-np.array(trans))
   elif id is None and prev_trans is None:
      prev_trans = trans

   if (id is None and dist > 0.03):
      print "BOX MOVED"
      next_id += 1
      prev_trans = translation

   if id is None:
      marker_box.id = next_id
   else:
      marker_box.id = id

   marker_box.lifetime = lifetime

   marker_box.header.frame_id = "base"
   marker_box.type = marker_box.CUBE
   marker_box.ns = ns
   marker_box.action = Marker.ADD
   marker_box.scale.x = box_length
   marker_box.scale.y = box_height
   marker_box.scale.z = box_breadth
   marker_box.color.a = a
   marker_box.color.r = r
   marker_box.color.g = g
   marker_box.color.b = b
   marker_box.pose.orientation.w = ori[3]
   marker_box.pose.orientation.x = ori[0]
   marker_box.pose.orientation.y = ori[1]
   marker_box.pose.orientation.z = ori[2]
   marker_box.pose.position.x    =  trans[0]
   marker_box.pose.position.y    =  trans[1]
   marker_box.pose.position.z    =  trans[2] - box_height/2

   return marker_box

print "Sleeping 2 seconds"
rospy.sleep(2)

while not rospy.is_shutdown():

   markerArray = MarkerArray()

  
   try:
    time = box_tf.getLatestCommonTime('base', 'box')
    time2 = box_tf.getLatestCommonTime('base', 'box_goal')
    flag_box = True
   except tf.Exception:
      print "Some exception occured!!!"

   if flag_box:
      flag_box = False
      translation, ori = box_tf.lookupTransform('base', 'box', time)

      translation_goal, ori_goal = box_tf.lookupTransform('base', 'box_goal', time2)

      marker_box_traj = make_box_marker(trans = translation, ori = ori, id = None, r = 0.5, g = 1.0, b = 1.0, a = 1.0, ns = 'box_traj')

      marker_box = make_box_marker(trans = translation, ori = ori, id = 0, r = 0.5, g = 1.0, b = 0.1, a = 1.0)
      

      marker_box_goal = make_box_marker(trans = translation_goal, ori = ori_goal, id = 1, r = 1.0, g = 0.1, b = 0.1, a = 1.0, lifetime = rospy.Duration(1))

      marker_table = Marker()
      marker_table.id = 2
      marker_table.action = Marker.ADD;
      marker_table.type = Marker.MESH_RESOURCE;
      # marker_table.mesh_use_embedded_materials = True

      marker_table.color.a = 0.7
      marker_table.color.r = 0.7
      marker_table.color.g = 0.7
      marker_table.color.b = 0.7

      marker_table.pose.orientation.w = 1
      marker_table.pose.orientation.x = 0
      marker_table.pose.orientation.y = 0
      marker_table.pose.orientation.z = 0
      marker_table.pose.position.x    = 0.737279182646236
      marker_table.pose.position.y    = 0.30313784831064317
      marker_table.pose.position.z    = -0.55229382998687607
      marker_table.scale.x = 1.0
      marker_table.scale.y = 2.0
      marker_table.scale.z = 0.5
      marker_table.header.frame_id = "base"
      marker_table.mesh_resource = 'package://sawyer_sim_examples/models/cafe_table/meshes/cafe_table.dae'
      

      markerArray.markers.append(marker_box_traj)
      markerArray.markers.append(marker_table)
      markerArray.markers.append(marker_box_goal)
      markerArray.markers.append(marker_box)



   # Publish the MarkerArray
   publisher.publish(markerArray)


   rate.sleep()