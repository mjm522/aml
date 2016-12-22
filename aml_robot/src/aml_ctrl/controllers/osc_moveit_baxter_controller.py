import sys
import copy
import rospy
import geometry_msgs.msg

from std_msgs.msg import (Header, String)
from geometry_msgs.msg import (PoseStamped, Pose, Point, Quaternion)

import moveit_commander
from moveit_msgs.msg import DisplayTrajectory, CollisionObject
from shape_msgs.msg import SolidPrimitive

import numpy as np
import quaternion

from moveit_commander import MoveGroupCommander

#instructions:
#open 3 terminals
#terminal 1: source baster.sh, rosrun baxter_interface joint_trajectory_action_server.py
#terminal 2: source baster.sh, roslaunch baxter_moveit_config demo_baxter.launch
#terminal 3: source baster.sh, run this file

class BaxterMoveItController():

    def __init__(self):
        #First initialize moveit_commander and rospy.
        moveit_commander.roscpp_initialize(sys.argv)
        
        #Instantiate a RobotCommander object. This object is an interface to the robot as a whole.
        self.robot = moveit_commander.RobotCommander()

        #Instantiate a PlanningSceneInterface object. This object is an interface to the world surrounding the robot.
        try:
            self.scene = moveit_commander.PlanningSceneInterface()
            #Wait for RVIZ to initialize. This sleep is ONLY to allow Rviz to come up.
            rospy.sleep(0.5)
        except Exception as e:
            raise e

        self.left_group_configured  = False
        self.right_group_configured = False
        self.both_group_configured  = False

        #We create this DisplayTrajectory publisher which is used below to publish trajectories for RVIZ to visualize.
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', DisplayTrajectory)

    def add_static_objects_to_scene(self, limb_group=0, obj_pos=None, obj_ori=None):
        #TODO: this funciton is buggy, will crash, need to fix it, if needed

        collision_object = CollisionObject()
        
        #get handle of the move group
        # move_group =  self.get_group_handle(limb_group=limb_group)
        # move_group = self.scene.MoveGroupInterface('left_arm')

        
        # collision_object.header.frame_id = moveit_commander.getPlanningFrame()
        
        #id of the object is used to identify it
        collision_object.id = "box1"
        
        #define a box to add to the world
        primitive = SolidPrimitive()
        primitive.type = primitive.BOX
        primitive.dimensions.resize(3)
        primitive.dimensions[0] = 0.4
        primitive.dimensions[1] = 0.1
        primitive.dimensions[2] = 0.4

        #define a pose for the box
        box_pose = Pose()
        box_pose.orientation.w = 1.0
        box_pose.position.x = 0.6
        box_pose.position.y = -0.4
        box_pose.position.z = 1.2

        # collision_object.primitives.push_back(primitive)
        # collision_object.primitive_poses.push_back(box_pose)
        # collision_object.operation = collision_object.ADD
        add_box()
        #add the collision object into the world
        # self.scene.addCollisionObjects(collision_object)
        #Sleep to allow MoveGroup to recieve and process the collision object message
        rospy.sleep(1.0)


    def set_group_handles(self, limb_group=1):

        #0 for left arm handle, 
        #1 for right arm handle
        #2 for both arm handle

        if limb_group==0:
            self.group_left_arm = MoveGroupCommander("left_arm")
            self.set_tolerance(self.group_left_arm)
            self.left_group_configured = True

        if limb_group==1:
            self.group_right_arm = MoveGroupCommander("right_arm")
            self.set_tolerance(self.group_right_arm)
            self.right_group_configured = True

        if limb_group==2:
            self.group_both_arms = MoveGroupCommander("both_arms")
            self.set_tolerance(self.group_both_arms)
            self.both_group_configured = True


    def set_tolerance(self, group_handle, pos_tol=0.01, ori_tol=0.01):
        group_handle.set_goal_position_tolerance(pos_tol)
        group_handle.set_goal_orientation_tolerance(ori_tol)

    def get_group_handle(self, limb_group=1):
        #0 for left arm handle, 
        #1 for right arm handle
        #2 for both arm group_handle

        if (limb_group==0) and self.left_group_configured:
            return self.group_left_arm

        elif (limb_group==1) and self.right_group_configured:
            return self.group_right_arm

        elif (limb_group==2) and self.both_group_configured:
            return self.group_both_arms
        
        else:
            print "Trying to get unconfigured handle, call set_group_handle first"
            raise ValueError


    def get_plan(self, limb_group, pos, ori, wait_time=1.5):
        arm_handle = self.get_group_handle(limb_group=limb_group)

        if isinstance(ori, np.quaternion):
            ori = quaternion.as_float_array(ori)[0]

        pose_target = geometry_msgs.msg.Pose()
        pose_target.orientation.x = ori[1]
        pose_target.orientation.y = ori[2]
        pose_target.orientation.z = ori[3]
        pose_target.orientation.w = ori[0]
        pose_target.position.x = pos[0]
        pose_target.position.y = pos[1]
        pose_target.position.z = pos[2]
        arm_handle.set_pose_target(pose_target)

        plan = arm_handle.plan()
        rospy.sleep(wait_time)

        return plan

    def execute_plan(self, limb_group, plan, real_robot=False):
        arm_handle = self.get_group_handle(limb_group)
        
        if real_robot:
            arm_handle.go(wait=True)
        
        arm_handle.execute(plan)

    def clear_pose_targets(self, limb_group=1):

        if (limb_group==0) and self.left_group_configured:
            self.group_left_arm.clear_pose_targets()

        if (limb_group==1) and self.right_group_configured:
            self.group_right_arm.clear_pose_targets()

        if (limb_group==2) and self.both_group_configured:
            self.group_both_arms.clear_pose_targets()
 
    def clean_shutdown(self):

        self.clear_pose_targets(limb_group=1)

        print "Moveit shutting down..."

        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)

    def plan_both_arms(self, left_pos, left_ori, right_pos, right_ori):
        arm_handle = self.get_group_handle(3)
        # group_both_arms.set_pose_target(pose_target_right, 'right_gripper')
        # group_both_arms.set_pose_target(pose_target_left, 'left_gripper')
        # plan_both = group_both_arms.plan()

        return plan

    def self_test(self, limb_group=1):
        #these are two valid poses that can to do self test
        if limb_group==0:
            pos = [0.81576, 0.093893, 0.2496]
            ori = [0.67253, 0.69283, 0.1977, -0.16912]   

        if limb_group==1:
            pos = [ 0.72651, -0.041037, 0.19097]
            ori = [-0.33955, 0.56508, -0.5198, -0.54332]

        self.set_group_handles(limb_group=limb_group)
        plan = self.get_plan(limb_group=limb_group, pos=pos, ori=ori)

        self.execute_plan(limb_group=limb_group, plan=plan, real_robot=True)
        self.clean_shutdown()
 

if __name__ == '__main__':
    rospy.init_node('baxter_moveit',anonymous=True)
    baxter_ctrlr = BaxterMoveItController()
    limb_group = 1
    main(limb_group)
    baxter_ctrlr.self_test(limb_group=limb_group)
