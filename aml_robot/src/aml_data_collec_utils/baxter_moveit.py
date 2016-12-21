import sys
import copy
import rospy
import geometry_msgs.msg

from std_msgs.msg import (Header, String)
from geometry_msgs.msg import (PoseStamped, Pose, Point, Quaternion)

import moveit_commander
import moveit_msgs.msg

from moveit_commander import MoveGroupCommander

#instructions:
#open 3 terminals
#terminal 1: source baster.sh, rosrun baxter_interface joint_trajectory_action_server.py
#terminal 2: source baster.sh, roslaunch baxter_moveit_config demo_baxter.launch
#terminal 3: source baster.sh, run this file

class BaxterMoveItController():

    def __init__(self):
        #First initialize moveit_commander and rospy.
        self.moveit_commander.roscpp_initialize(sys.argv)
        
        #Instantiate a RobotCommander object. This object is an interface to the robot as a whole.
        self.robot = moveit_commander.RobotCommander()

        #Instantiate a PlanningSceneInterface object. This object is an interface to the world surrounding the robot.
        try:
            self.scene = moveit_commander.PlanningSceneInterface()
            #Wait for RVIZ to initialize. This sleep is ONLY to allow Rviz to come up.
            rospy.sleep(0.5)
        except Exception as e:
            raise e

        self.left_group_configured = False
        self.right_group_configured = False
        self.both_group_configured = False

        #We create this DisplayTrajectory publisher which is used below to publish trajectories for RVIZ to visualize.
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory)


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
        #2 for both arm handle
        if limb_group==0 and self.left_group_configured:
            return self.group_left_arm
        else:
            print "Trying to get unconfigured handle, call set_group_handle first"
            raise ValueError

        if limb_group==1 and self.right_group_configured:
            return self.group_right_arm
        else:
            print "Trying to get unconfigured handle, call set_group_handle first"
            raise ValueError

        if limb_group==2 and self.both_group_configured:
            return self.group_both_arms
        else:
            print "Trying to get unconfigured handle, call set_group_handle first"
            raise ValueError


    def get_plan(self, limb_group, pos, ori, wait_time=1.5):
        arm_handle = self.get_group_handle(limb_group)
        pose_target = geometry_msgs.msg.Pose()
        pose_target.orientation.x = ori[1]
        pose_target.orientation.y = ori[2]
        pose_target.orientation.z = ori[3]
        pose_target.orientation.w = ori[0]
        pose_target.position.x = pos[0]
        pose_target.position.y = pos[1]
        pose_target.position.z = pos[2]
        arm_handle.set_pose_target(pose_target_left)

        plan = arm_handle.plan()
        rospy.sleep(wait_time)

        return plan

    def execute_plan(self, limb_group, plan, real_robot=False):
        arm_handle = self.get_group_handle(limb_group)
        
        if real_robot:
            arm_handle.go(wait=True)
        
        arm_handle.execute(plan)

    def clear_pose_targets(self, limb_group=1):

        if limb_group==0 and self.left_group_configured:
            self.group_left_arm.clear_pose_targets()
        else:
            print "Trying to get unconfigured handle, call set_group_handle first"
            raise ValueError

        if limb_group==1 and self.right_group_configured:
            self.group_right_arm.clear_pose_targets()
        else:
            print "Trying to get unconfigured handle, call set_group_handle first"
            raise ValueError

        if limb_group==2 and self.both_group_configured:
            self.group_both_arms.clear_pose_targets()
        else:
            print "Trying to get unconfigured handle, call set_group_handle first"
            raise ValueError

    def clean_shutdown(self):

        self.clear_pose_targets(limb_group=1)

        print "Moveit shutting down..."

        self.moveit_commander.roscpp_shutdown()
        self.moveit_commander.os._exit(0)

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

        self.execute_plan(limb_group=limb_group, plan=plan, real_robot=False)
        self.clean_shutdown()
     

if __name__ == '__main__':
    rospy.init_node('baxter_moveit',anonymous=True)
    baxter_ctrlr = BaxterMoveItController()
    limb_group = 1
    baxter_ctrlr.self_test(limb_group=limb_group)


# #First initialize moveit_commander and rospy.
# moveit_commander.roscpp_initialize(sys.argv)
# rospy.init_node('baxter_moveit',anonymous=True)

# #Instantiate a RobotCommander object. This object is an interface to the robot as a whole.
# robot = moveit_commander.RobotCommander()

# #Wait for RVIZ to initialize. This sleep is ONLY to allow Rviz to come up.
# print "Waiting for RVIZ..."
# rospy.sleep(2)

# #Instantiate a PlanningSceneInterface object. This object is an interface to the world surrounding the robot.
# scene = moveit_commander.PlanningSceneInterface()

# #Instantiate a MoveGroupCommander object. This object is an interface to one group of joints. In this case the group is the joints in the left arm. This interface can be used to plan and execute motions on the left arm.
# group_both_arms = MoveGroupCommander("both_arms")
# group_both_arms.set_goal_position_tolerance(0.01)
# group_both_arms.set_goal_orientation_tolerance(0.01)

# group_left_arm = MoveGroupCommander("left_arm")
# group_left_arm.set_goal_position_tolerance(0.01)
# group_left_arm.set_goal_orientation_tolerance(0.01)

# group_right_arm = MoveGroupCommander("right_arm")
# group_right_arm.set_goal_position_tolerance(0.01)
# group_right_arm.set_goal_orientation_tolerance(0.01)

# #We create this DisplayTrajectory publisher which is used below to publish trajectories for RVIZ to visualize.
# display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory)



# #Call the planner to compute the plan and visualize it if successful.
# print "Generating plan for left arm"
# pose_target_left = geometry_msgs.msg.Pose()
# pose_target_left.orientation.x = 0.69283
# pose_target_left.orientation.y = 0.1977
# pose_target_left.orientation.z = -0.16912
# pose_target_left.orientation.w = 0.67253
# pose_target_left.position.x = 0.81576
# pose_target_left.position.y = 0.093893
# pose_target_left.position.z = 0.2496
# group_left_arm.set_pose_target(pose_target_left)

# #group_left_arm.set_position_target([0.75,0.27,0.35])
# plan_1eft = group_left_arm.plan()
# #print "Trajectory time (nsec): ", plan_left.joint_trajectory.points[len(plan_left.joint_trajectory.points)-1].time_from_start

# rospy.sleep(2)
# print "Generating plan for right arm"
# pose_target_right = geometry_msgs.msg.Pose()
# pose_target_right.orientation.x = 0.56508
# pose_target_right.orientation.y = -0.5198
# pose_target_right.orientation.z = -0.54332
# pose_target_right.orientation.w = -0.33955
# pose_target_right.position.x = 0.72651
# pose_target_right.position.y = -0.041037
# pose_target_right.position.z = 0.19097
# group_right_arm.set_pose_target(pose_target_right)

# #group_right_arm.set_position_target([0.75,-0.27,0.35])
# plan_right = group_right_arm.plan()
# #print "Trajectory time (nsec): ", plan_right.joint_trajectory.points[len(plan_right.joint_trajectory.points)-1].time_from_start
# rospy.sleep(2)

# # Uncomment below line when working with a real robot
# # group.go(wait=True)

# group_left_arm.execute(plan_1eft)
# group_right_arm.execute(plan_right)

# group_left_arm.clear_pose_targets()
# group_right_arm.clear_pose_targets()

# rospy.sleep(7)
# print "going to shut down"

# moveit_commander.roscpp_shutdown()
# moveit_commander.os._exit(0)

# print "Generating plan for both arms"
# #group_both_arms.set_pose_target(pose_target_right, pose_target_left)
# group_both_arms.set_pose_target(pose_target_right, 'right_gripper')
# group_both_arms.set_pose_target(pose_target_left, 'left_gripper')

# plan_both = group_both_arms.plan()

# print "Trajectory time (nsec): ", plan_both.joint_trajectory.points[len(plan_both.joint_trajectory.points)-1].time_from_start