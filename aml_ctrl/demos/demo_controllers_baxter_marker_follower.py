#!/usr/bin/env python

# USAGE: rosrun rosrun aml_ctrl demo_controller_baxter_sim_follow_marker <controller_id>
# where <controller_id> is (default:1):
# 1 (Position Control), 2 (Velocity Control), or 3 (Torque Control)

"""
This demo is for showing various contollers in operations space.
The demo opens up a marker in the rviz environment, which can be
dragged to places (and rotated) to give a setpoint. The type of controller and the arm to 
be contolled can be choosen by righ clicking in the rviz environment.
By default position controller is chosen.
Another variable called multiple_goals (defualt value=0) is used to help the robot move along a trajectory
rather than a fixed goal. If this parameter is set (i.e. multiple_goals=1), seires of position and
orientation rising from the mouse movement is recorded and played when the move button is lifted. 
"""

import sys
import rospy
import argparse
import quaternion
import numpy as np

from geometry_msgs.msg import Point
from visualization_msgs.msg import *
from interactive_markers.menu_handler import *
from interactive_markers.interactive_marker_server import *

from aml_robot.baxter_robot import BaxterArm
from aml_visual_tools.rviz_markers import RvizMarkers
from aml_ctrl.controllers.os_controllers.os_torque_controller import OSTorqueController
from aml_ctrl.controllers.os_controllers.os_postn_controller import OSPositionController
from aml_ctrl.controllers.os_controllers.os_velocity_controller import OSVelocityController
from aml_ctrl.controllers.os_controllers.os_impedance_controller import OSImpedanceController

menu_handler      = MenuHandler()
destinationMarker = RvizMarkers()


global multiple_goals

set_points_pos = []
set_points_ori = []

def process_feedback(feedback):
    """
    Feed back callback function
    """

    global set_points_pos, set_points_ori, multiple_goals
    
    p = feedback.pose.position
    q = feedback.pose.orientation

    print feedback.marker_name + "_marker is now at " + str(p.x) + ", " + str(p.y) + ", " + str(p.z)

    goal_pos = np.array([p.x,p.y,p.z])
    goal_ori = np.quaternion(q.w, q.x,q.y,q.z)

    if multiple_goals:
        set_points_pos.append(goal_pos)
        set_points_ori.append(goal_ori)
    else:
        set_points_pos = [goal_pos]
        set_points_ori = [goal_ori]

    if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:

        ctrlr.set_active(True)

        print len(set_points_pos)

        for set_pos, set_ori in zip(set_points_pos, set_points_ori):

            ctrlr.set_goal(goal_pos=set_pos, 
                           goal_ori=set_ori, 
                           orientation_ctrl = True)

            lin_error, ang_error, success, time_elapsed = ctrlr.wait_until_goal_reached(timeout=1.0)

        set_points = []

    rate.sleep()

def switch_arm(armfeedback):
    """
    Function to select the arm
    """
    global ctrlr, limb
    if armfeedback.menu_entry_id == 2:
        limb = r_limb
        print 'Right arm selected'
    elif armfeedback.menu_entry_id == 3:
        limb = l_limb
        print "Left arm selected"

    ctrlr = set_controller(control_id, limb)
    ctrlr.set_active(True)

def set_controller(controller_id, arm):
    """
    Function to choose a controller
    """
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

def switch_controller(switch_control_feedback):
    """
    callback to choose the contoller
    """
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
    ctrlr = set_controller(control_id,limb)
    ctrlr.set_active(True)

def init_menu():
    arm_menu = menu_handler.insert("Arm")
    menu_handler.insert( "Right arm", parent = arm_menu, callback=switch_arm )
    menu_handler.insert( "Left arm", parent = arm_menu, callback=switch_arm )

    control_menu = menu_handler.insert("Controller")
    menu_handler.insert("Position Controller", parent = control_menu, callback =  switch_controller)
    menu_handler.insert("Velocity Controller", parent = control_menu, callback =  switch_controller)
    menu_handler.insert("Torque Controller", parent = control_menu, callback =  switch_controller)
    menu_handler.insert("Impedance Controller", parent = control_menu, callback = switch_controller)

if __name__=="__main__":
    
    #taking the params using rospy since 
    #ros passes some additional parameters to the file when run from a launch file
    #hence sys.args will crash

    global multiple_goals

    parser = argparse.ArgumentParser(description='Collect demonstrations')

    parser.add_argument('-c', '--control_id', type=int, default=1, help='type of controller-(1. Pos 2. Vel 3. Torque 4. Impedance)')

    parser.add_argument('-m', '--multiple_goals', type=int, default=0, help='should follow multiple goals')
    
    args = parser.parse_args(rospy.myargv()[1:])

    control_id = args.control_id
    multiple_goals = args.multiple_goals

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
    ctrlr = set_controller(control_id, limb)

    start_pos, start_ori = limb.get_ee_pose()
    goal_ori = start_ori#quaternion.as_float_array(start_ori)
    ctrlr.set_active(True)

    # create an interactive marker server on the topic namespace basic_control
    server = InteractiveMarkerServer("basic_control")
    rate = rospy.Rate(100)

    # create an interactive marker for our server
    position = Point( start_pos[0], start_pos[1], start_pos[2])
    marker = destinationMarker.makeMarker( False, InteractiveMarkerControl.MOVE_ROTATE_3D, position, quaternion.as_float_array(start_ori), True)
    server.insert(marker, process_feedback)
    menu_handler.apply( server, marker.name )

    # 'commit' changes and send to all clients
    server.applyChanges()

    rospy.spin()