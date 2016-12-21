import sys
import copy
import rospy
import geometry_msgs.msg

from std_msgs.msg import (Header, String)
from geometry_msgs.msg import (PoseStamped, Pose, Point, Quaternion)

import moveit_commander
import moveit_msgs.msg

from moveit_commander import MoveGroupCommander


class BaxterController(object):
    def __init__(self):
        # ...
        self.commander = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.right_arm_moveit = MoveGroupCommander("right_arm")
        self.right_arm_moveit.set_goal_position_tolerance(0.01)
        # ...

    def get_plan(self, points, orientation, limb):
        """Get smooth plan through given list of points
        orientation is a geometry_msgs.msg.Quaternion
        """
        group = self.get_arm(limb) # returns the appropriate MoveGroupCommander
        now = group.get_current_pose().pose
        waypoints = [now]
        for x, y, z in points:
            x, y, z = self.get_adjusted_goal(x, y, z, limb) # small constant offsets
            p = util.get_pose_stamped(x, y, z, O)
            waypoints.append(p)
        plan, f = group.compute_cartesian_path(waypoints, 0.01, 0.0)
        # f is the fraction of succesfully planned path
        return plan, f

    def slow_plan(self, plan):
        """Explicitly sets plan velocities, use if you see velocity errors"""
        n = 7
        vels = [0.1] * n
        for p in plan.joint_trajectory.points:
            p.velocities = vels

    def do_plan(self, plan, limb = None):
        group = self.get_arm(limb)
        
return group.execute(plan)