#!/usr/bin/env python

# USAGE: rosrun rosrun aml_ctrl demo_controller_baxter_sim_trajectory_tracker <controller_id>
# where <controller_id> is (default:1):
# 1 (Position Control), 2 (Velocity Control), or 3 (Torque Control)

import rospy
import sys

from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from geometry_msgs.msg import Point

from aml_robot.baxter_robot import BaxterArm
from aml_ctrl.controllers.os_controllers.os_postn_controller import OSPositionController
from aml_ctrl.controllers.os_controllers.os_torque_controller import OSTorqueController
from aml_ctrl.controllers.os_controllers.os_velocity_controller import OSVelocityController
from aml_ctrl.controllers.os_controllers.os_impedance_controller import OSImpedanceController

from aml_visual_tools.rviz_markers import RvizMarkers

import numpy as np
import quaternion

menu_handler = MenuHandler()
destinationMarker = RvizMarkers()

set_points = []

def processFeedback(feedback):
    global set_points
    p = feedback.pose.position

    print feedback.marker_name + "_marker is now at " + str(p.x) + ", " + str(p.y) + ", " + str(p.z)

    goal_pos = np.array([p.x,p.y,p.z])

    set_points.append(goal_pos)

    if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:

        ctrlr.set_active(True)
        for set_point in set_points:


            ctrlr.set_goal(goal_pos=set_point, 
                           goal_ori=goal_ori, 
                           orientation_ctrl = False)

            lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=1.0)

        set_points = []

    rate.sleep()

def switchArm(armfeedback):
    global ctrlr, limb
    if armfeedback.menu_entry_id == 2:
        limb = r_limb
        print 'Right arm selected'
    elif armfeedback.menu_entry_id == 3:
        limb = l_limb
        print "Left arm selected"

    ctrlr = setController(control_id, limb)
    ctrlr.set_active(True)

def setController(controller_id, arm):
    global controller_defined, ctrlr
    if controller_defined:
        ctrlr.set_active(False)

    if controller_id == 1:
        controller = OSPositionController(arm)
    elif controller_id == 2:
        controller = OSVelocityController(arm)
    elif controller_id == 3:
        controller = OSTorqueController(arm)
    elif controller_id == 4:
        controller = OSImpedanceController(arm)

    controller_defined = True
    return controller

def switchController(switch_control_feedback):
    global limb, ctrlr, control_id
    if switch_control_feedback.menu_entry_id == 5:
        print "Switching to Position Control"
        control_id = 1
    if switch_control_feedback.menu_entry_id == 6:
        print "Switching to Velocity Control"
        control_id = 2
    if switch_control_feedback.menu_entry_id == 7:
        print "Switching to Torque Control"
        control_id = 3
    if switch_control_feedback.menu_entry_id == 8:
        print "Switching to Impedance Control"
        control_id = 4
    ctrlr = setController(control_id,limb)
    ctrlr.set_active(True)

def init_menu():
    arm_menu = menu_handler.insert("Arm")
    menu_handler.insert( "Right arm", parent = arm_menu, callback=switchArm )
    menu_handler.insert( "Left arm", parent = arm_menu, callback=switchArm )

    control_menu = menu_handler.insert("Controller")
    menu_handler.insert("Position Controller", parent = control_menu, callback =  switchController)
    menu_handler.insert("Velocity Controller", parent = control_menu, callback =  switchController)
    menu_handler.insert("Torque Controller", parent = control_menu, callback =  switchController)
    menu_handler.insert("Impedance Controller", parent = control_menu, callback = switchController)

if __name__=="__main__":
    if not len(sys.argv) == 2:
        if len(sys.argv) == 1:
            control_id = 1
            print "Using Position Control"
        else:
            print "\nUSAGE: rosrun aml_ctrl demo_controller_baxter_sim_follow_marker <controller_id>\n\nwhere <controller_id> is (default:1):\n1. Position Control\n2. Velocity Control\n3. Torque Control\n3. Impedance Control\n"
            sys.exit()
    else:
        if float(sys.argv[1]) < 5 and float(sys.argv[1]) > 0:
            control_id = float(sys.argv[1])
            if control_id == 1:
                print "Using Position Controller"
            elif control_id ==2:
                print "Using Velocity Controller"
            elif control_id == 3:
                print "Using Torque Controller"
            elif control_id == 4:
                print "Using Impedance Controller"
        else:
            print "\nInvalid Control ID!"
            print "\nUSAGE: rosrun aml_ctrl demo_controller_baxter_sim_follow_marker <controller_id>\n\nwhere <controller_id> is (default:1):\n1. Position Control\n2. Velocity Control\n3. Torque Control\n3. Impedance Control\n"
            sys.exit()

    controller_defined = False

    rospy.init_node("track_3d_marker_gravity_comp")
    init_menu()

    r_limb = BaxterArm('right')
    l_limb = BaxterArm('left')
    r_limb.untuck_arm()
    l_limb.untuck_arm()

    limb = r_limb
    ctrlr = setController(control_id, limb)

    start_pos, start_ori = limb.get_ee_pose()
    goal_ori = start_ori

    ctrlr.set_active(True)

    # create an interactive marker server on the topic namespace basic_control
    server = InteractiveMarkerServer("basic_control")
    rate = rospy.Rate(100)

    # create an interactive marker for our server
    position = Point( start_pos[0], start_pos[1], start_pos[2])
    marker = destinationMarker.makeMarker( False, InteractiveMarkerControl.MOVE_ROTATE_3D, position, quaternion.as_float_array(start_ori), True)
    server.insert(marker, processFeedback)
    menu_handler.apply( server, marker.name )

    # 'commit' changes and send to all clients
    server.applyChanges()

    rospy.spin()
