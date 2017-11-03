#!/usr/bin/env python

import rospy
# import copy

from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from geometry_msgs.msg import Point
from tf.broadcaster import TransformBroadcaster

import baxter_interface
import baxter_external_devices

from aml_robot import baxter_robot

from std_msgs.msg import (
    UInt16,
)

from baxter_interface import CHECK_VERSION

# from aml_perception import camera_sensor
# from functools import partial


import numpy as np
import quaternion

# from random import random
# from math import sin

# server = None
menu_handler = MenuHandler()
# br = None
# counter = 0

def processFeedback(feedback):
    p = feedback.pose.position
    print feedback.marker_name + "_marker is now at " + str(p.x) + ", " + str(p.y) + ", " + str(p.z)

    goal_ori = quaternion.as_float_array(start_ori)
    goal_pos = np.array([p.x,p.y,p.z])
    current_ee_pos, current_ee_ori = limb.get_ee_pose()

    success, goal_joint_angles = limb.ik(goal_pos,goal_ori)
    limb.exec_position_cmd(goal_joint_angles)
    rate.sleep()

def makeBox( msg ):
    marker = Marker()

    marker.type = Marker.CUBE
    marker.scale.x = msg.scale * 0.2
    marker.scale.y = msg.scale * 0.2
    marker.scale.z = msg.scale * 0.2
    marker.color.r = 0.5
    marker.color.g = 0.5
    marker.color.b = 0.5
    marker.color.a = 1.0

    return marker

def makeBoxControl( msg ):
    control =  InteractiveMarkerControl()
    control.always_visible = True
    control.markers.append( makeBox(msg) )
    msg.controls.append( control )
    return control

def makeMarker( fixed, interaction_mode, position, show_6dof = False):
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = "base"
    int_marker.pose.position = position
    int_marker.scale = 1

    int_marker.name = "simple_control"
    int_marker.description = "Simple Control"

    # insert a box
    makeBoxControl(int_marker)
    int_marker.controls[0].interaction_mode = interaction_mode

    if fixed:
        int_marker.name += "_fixed"
        int_marker.description += "\n(fixed orientation)"

    if interaction_mode != InteractiveMarkerControl.NONE:
        control_modes_dict = { 
                          InteractiveMarkerControl.MOVE_3D : "MOVE_3D",
                          InteractiveMarkerControl.ROTATE_3D : "ROTATE_3D",
                          InteractiveMarkerControl.MOVE_ROTATE_3D : "MOVE_ROTATE_3D" }
        int_marker.name += "_" + control_modes_dict[interaction_mode]
        int_marker.description = "3D Control"
        if show_6dof: 
          int_marker.description += " + 6-DOF controls"
        int_marker.description += "\n" + control_modes_dict[interaction_mode]
    
    if show_6dof: 
        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        control.name = "rotate_x"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        if fixed:
            control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        control.name = "move_x"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        if fixed:
            control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        control.name = "rotate_z"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        if fixed:
            control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        control.name = "move_z"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        if fixed:
            control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 1
        control.name = "rotate_y"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        if fixed:
            control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 1
        control.name = "move_y"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        if fixed:
            control.orientation_mode = InteractiveMarkerControl.FIXED
        int_marker.controls.append(control)
    # server.insert()
    
    return int_marker
    # print 'here'

def switchArm(armfeedback):
    global limb
    # print armfeedback.menu_entry_id
    if armfeedback.menu_entry_id == 1:
        limb = r_limb
        print 'Right arm selected'
    elif armfeedback.menu_entry_id == 2:
        limb = l_limb
        print "Left arm selected"

# if __name__=="__main__":
rospy.init_node("track_3d_marker")

_rs = baxter_interface.RobotEnable(CHECK_VERSION)
_rs.enable()    

r_limb = baxter_robot.BaxterArm('right')
l_limb = baxter_robot.BaxterArm('left')
r_limb.untuck_arm()
l_limb.untuck_arm()

limb = r_limb

menu_handler.insert( "Right arm", callback=switchArm )
menu_handler.insert( "Left arm", callback=switchArm )

start_pos, start_ori = limb.get_ee_pose()

# create an interactive marker server on the topic namespace basic_control
server = InteractiveMarkerServer("basic_control")
rate = rospy.Rate(5)

# create an interactive marker for our server
position = Point( 3, 0, 0)
marker = makeMarker( False, InteractiveMarkerControl.MOVE_3D, position, False)
server.insert(marker, processFeedback)
menu_handler.apply( server, marker.name )
# print "Marker is now at ", marker.pose.position

# 'commit' changes and send to all clients
server.applyChanges()

rospy.spin()
