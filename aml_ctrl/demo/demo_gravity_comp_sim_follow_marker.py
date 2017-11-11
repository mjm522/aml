#!/usr/bin/env python

import rospy
# import copy

from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from geometry_msgs.msg import Point
from tf.broadcaster import TransformBroadcaster

from aml_robot.baxter_robot import BaxterArm
from aml_ctrl.utilities.min_jerk_interp import MinJerkInterp
from aml_ctrl.controllers.os_controllers.os_torque_controller import OSTorqueController
from aml_ctrl.controllers.os_controllers.os_velocity_controller import OSVelocityController
from aml_ctrl.controllers.os_controllers.os_postn_controller import OSPositionController



import numpy as np
import quaternion

# from random import random
# from math import sin

# server = None
menu_handler = MenuHandler()
# br = None
# counter = 0

def processFeedback(feedback):
    global start_ori, start_pos
    p = feedback.pose.position

    ctrlr.set_active(False)
    print feedback.marker_name + "_marker is now at " + str(p.x) + ", " + str(p.y) + ", " + str(p.z)

    goal_ori = quaternion.as_float_array(start_ori)
    goal_pos = np.array([p.x,p.y,p.z])

    if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:

        ctrlr.set_active(True)
        min_jerk_interp.configure(start_pos=start_pos, goal_pos=goal_pos, start_qt=start_ori, goal_qt=goal_ori)

        min_jerk_traj = min_jerk_interp.get_interpolated_trajectory()

        # ctrlr.set_goal(goal_pos,goal_ori)

        ctrlr.set_goal(goal_pos=goal_pos, 
                       goal_ori=goal_ori, 
                       orientation_ctrl = False)

        lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=1.0)

        start_pos = goal_pos
        start_ori = goal_ori
    # ctrlr.set_active(False)
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

    int_marker.name = "marker"
    int_marker.description = "Marker Control"

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
    global limb, ctrlr
    # print armfeedback.menu_entry_id
    if armfeedback.menu_entry_id == 1:
        limb = r_limb
        ctrlr = OSTorqueController(limb)
        print 'Right arm selected'
    elif armfeedback.menu_entry_id == 2:
        limb = l_limb
        ctrlr = OSTorqueController(limb)
        print "Left arm selected"



if __name__=="__main__":
    rospy.init_node("track_3d_marker_gravity_comp")


    r_limb = BaxterArm('right')
    l_limb = BaxterArm('left')
    r_limb.untuck_arm()
    l_limb.untuck_arm()


    limb = r_limb
    ctrlr = OSPositionController(limb)

    menu_handler.insert( "Right arm", callback=switchArm )
    menu_handler.insert( "Left arm", callback=switchArm )

    start_pos, start_ori = limb.get_ee_pose()
    goal_pos = np.array([0.005,-0.05,-0.05])
    goal_ori = start_ori

    min_jerk_interp = MinJerkInterp()

    ctrlr.set_active(True)

    # create an interactive marker server on the topic namespace basic_control
    server = InteractiveMarkerServer("basic_control")
    rate = rospy.Rate(100)

    # create an interactive marker for our server
    position = Point( start_pos[0], start_pos[1], start_pos[2])
    marker = makeMarker( False, InteractiveMarkerControl.MOVE_3D, position, False)
    server.insert(marker, processFeedback)
    menu_handler.apply( server, marker.name )

    # ctrlr.set_goal(goal_pos=goal_pos, 
    #                    goal_ori=goal_ori, 
    #                    orientation_ctrl = False)

    # while not rospy.is_shutdown():

    #     lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=1.0)
    # print "Marker is now at ", marker.pose.position

    # 'commit' changes and send to all clients
    server.applyChanges()

    rospy.spin()
