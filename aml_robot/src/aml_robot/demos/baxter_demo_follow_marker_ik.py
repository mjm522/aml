#!/usr/bin/env python

import rospy

from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from geometry_msgs.msg import Point

from aml_visual_tools.rviz_markers import RvizMarkers

from aml_robot import baxter_robot

import numpy as np
import quaternion

menu_handler = MenuHandler()
destinationMarker = RvizMarkers()

def processFeedback(feedback):
    p = feedback.pose.position
    print feedback.marker_name + "_marker is now at " + str(p.x) + ", " + str(p.y) + ", " + str(p.z)

    goal_ori = quaternion.as_float_array(start_ori)
    goal_pos = np.array([p.x,p.y,p.z])
    if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
        success, goal_joint_angles = limb.inverse_kinematics(goal_pos, goal_ori)
        limb.exec_position_cmd(goal_joint_angles)
    rate.sleep()

def switchArm(armfeedback):
    global limb
    if armfeedback.menu_entry_id == 1:
        limb = r_limb
        print 'Right arm selected'
    elif armfeedback.menu_entry_id == 2:
        limb = l_limb
        print "Left arm selected"

if __name__=="__main__":
    rospy.init_node("track_3d_marker_ik")

    r_limb = baxter_robot.BaxterArm('right')
    l_limb = baxter_robot.BaxterArm('left')
    r_limb.untuck()
    l_limb.untuck()

    limb = r_limb

    menu_handler.insert( "Right arm", callback=switchArm )
    menu_handler.insert( "Left arm", callback=switchArm )

    start_pos, start_ori = limb.ee_pose()

    # create an interactive marker server on the topic namespace basic_control
    server = InteractiveMarkerServer("basic_control")
    rate = rospy.Rate(100)

    # create an interactive marker for our server
    position = Point( start_pos[0], start_pos[1], start_pos[2])
    marker = destinationMarker.makeMarker( False, InteractiveMarkerControl.MOVE_3D, position, False)
    server.insert(marker, processFeedback)
    menu_handler.apply( server, marker.name )

    # 'commit' changes and send to all clients
    server.applyChanges()

    rospy.spin()
