#!/usr/bin/env python

import argparse
import struct
import sys
import copy

import rospy
import rospkg

from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import (
    Header,
    Empty,
)

from intera_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

import intera_interface
import intera_dataflow

WITH_GRIPPER=True

class PickAndPlace(object):
    def __init__(self, limb, hover_distance = 0.15, verbose=False):
        self._limb_name = limb # string
        self._head = intera_interface.Head()

        self._hover_distance = hover_distance # in meters
        self._verbose = verbose # bool
        self._limb = intera_interface.Limb(limb)
        if WITH_GRIPPER==True:
            self._gripper = intera_interface.Gripper(limb)
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = intera_interface.RobotEnable(intera_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        self._head.set_pan(0.0)

        self._joint_names = self._limb.joint_names()

        head_display = intera_interface.HeadDisplay()
        rospack = rospkg.RosPack()
        images_dir = rospack.get_path('intera_examples') + '/share/images/'
        head_display.display_image(images_dir + "sawyer_sdk_research.png", False, 1.0)

    def filtered_move_to_joint_positions(self, positions, timeout=15.0,
                                         threshold=0.008726646, test=None):
        """
        See baxter_interface.limb & intera_interface.limb
        """
        cmd = self._limb.joint_angles()

        def filtered_cmd():
            # First Order Filter - 0.2 Hz Cutoff
            for joint in positions.keys():
                cmd[joint] = 0.012488 * positions[joint] + 0.98751 * cmd[joint]
            return cmd

        def genf(joint, angle):
            def joint_diff():
                return abs(angle - self._limb._joint_angle[joint])
            return joint_diff

        diffs = [genf(j, a) for j, a in positions.items() if
                 j in self._limb._joint_angle]
        fail_msg = "{0} limb failed to reach commanded joint positions.".format(
                                                      self._limb.name.capitalize())

        def test_collision():
            if self._limb.has_collided():
                rospy.logerr(' '.join(["Collision detected.", fail_msg]))
                return True
            return False

        #self._limb.set_joint_positions(positions)
        self._limb.set_joint_positions(filtered_cmd())
        intera_dataflow.wait_for(
            test=lambda: test_collision() or \
                         (callable(test) and test() == True) or \
                         (all(diff() < threshold for diff in diffs)),
            timeout=timeout,
            timeout_msg=fail_msg,
            rate=100,
            raise_on_error=False,
            #body=lambda: self._limb.set_joint_positions(positions)
            body=lambda: self._limb.set_joint_positions(filtered_cmd())
            )

    def move_to_start(self, start_angles=None):
        # self._limb.set_joint_position_speed(0.0) # Not work with gazebo
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles)
        if WITH_GRIPPER==True:
            self.gripper_open()
        rospy.sleep(1.0)
        print("Running. Ctrl-c to quit")

    def ik_request(self, pose):
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            resp = self._iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False

        limb_joints = {}
        if (resp.result_type[0] != resp.IK_FAILED):
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            if self._verbose:
                print("IK Joint Solution:\n{0}".format(limb_joints))
                print("------------------")
        else:
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
            return False
        return limb_joints

    def _guarded_move_to_joint_position(self, joint_angles):
        if joint_angles:
            #self._limb.move_to_joint_positions(joint_angles)
            self.filtered_move_to_joint_positions(joint_angles)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

    def gripper_open(self):
        rospy.sleep(1.0)
        if WITH_GRIPPER==True:
            self._gripper.open()
        rospy.sleep(1.0)

    def gripper_close(self):
        rospy.sleep(1.0)
        if WITH_GRIPPER==True:
            self._gripper.close()
        rospy.sleep(1.0)

    def _approach(self, pose):
        approach = copy.deepcopy(pose)
        # approach with a pose the hover-distance above the requested pose
        approach.position.z = approach.position.z + self._hover_distance
        joint_angles = self.ik_request(approach)
        self._guarded_move_to_joint_position(joint_angles)

    def _retract(self):
        # retrieve current pose from endpoint
        current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x
        ik_pose.position.y = current_pose['position'].y
        ik_pose.position.z = current_pose['position'].z + self._hover_distance
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        joint_angles = self.ik_request(ik_pose)
        # servo up from current pose
        self._guarded_move_to_joint_position(joint_angles)

    def _servo_to_pose(self, pose):
        # servo down to release
        joint_angles = self.ik_request(pose)
        self._guarded_move_to_joint_position(joint_angles)

    def pick(self, pose):
        # open the gripper
        self.gripper_open()
        # servo above pose
        self._approach(pose)
        # servo to pose
        rospy.sleep(1.0)
        self._servo_to_pose(pose)
        # close gripper
        self.gripper_close()
        # retract to clear object
        self._retract()
        rospy.sleep(1.0)

    def place(self, pose):
        # servo above pose
        self._approach(pose)
        # servo to pose
        rospy.sleep(1.0)
        self._servo_to_pose(pose)
        # open the gripper
        self.gripper_open()
        # retract to clear object
        self._retract()
        rospy.sleep(1.0)

def load_gazebo_models(table_pose=Pose(position=Point(x=1.0, y=0.0, z=0.0)),
                       table_reference_frame="world",
                       block_pose=Pose(position=Point(x=0.7, y=0.15, z=0.7825)),
                       block_reference_frame="world"):
    # Get Models' Path
    model_path = rospkg.RosPack().get_path('sawyer_sim_examples')+"/models/"
    # Load Table SDF
    table_xml = ''
    with open (model_path + "cafe_table/model.sdf", "r") as table_file:
        table_xml=table_file.read().replace('\n', '')
    # Load Block URDF
    block_xml = ''
    with open (model_path + "block/model.urdf", "r") as block_file:
        block_xml=block_file.read().replace('\n', '')
    # Spawn Table SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_sdf = spawn_sdf("cafe_table", table_xml, "/",
                             table_pose, table_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Block URDF
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    try:
        spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        resp_urdf = spawn_urdf("block", block_xml, "/",
                               block_pose, block_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn URDF service call failed: {0}".format(e))

def delete_gazebo_models():
    # This will be called on ROS Exit, deleting Gazebo models
    # Do not wait for the Gazebo Delete Model service, since
    # Gazebo should already be running. If the service is not
    # available since Gazebo has been killed, it is fine to error out
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("cafe_table")
        resp_delete = delete_model("block")
    except rospy.ServiceException, e:
        rospy.loginfo("Delete Model service call failed: {0}".format(e))

def main():
    """RSDK Inverse Kinematics Pick and Place Example

    A Pick and Place example using the Rethink Inverse Kinematics
    Service which returns the joint angles a requested Cartesian Pose.
    This ROS Service client is used to request both pick and place
    poses in the /base frame of the robot.

    Note: This is a highly scripted and tuned demo. The object location
    is "known" and movement is done completely open loop. It is expected
    behavior that Intera will eventually mis-pick or drop the block. You
    can improve on this demo by adding perception and feedback to close
    the loop.
    """
    rospy.init_node("ik_pick_and_place_demo")
    # Load Gazebo Models via Spawning Services
    # Note that the models reference is the /world frame
    # and the IK operates with respect to the /base frame
    load_gazebo_models()
    # Remove models from the scene on shutdown
    rospy.on_shutdown(delete_gazebo_models)

    # Wait for the All Clear from emulator startup
    rospy.wait_for_message("/robot/sim/started", Empty)

    limb = 'right'
    hover_distance = 0.15 # meters
    # Starting Joint angles for left arm
    starting_joint_angles = {'right_j0': 0.00,
                            'right_j1': -1.18,
                            'right_j2': 0.0,
                            'right_j3': 2.18,
                            'right_j4': 0.00,
                            'right_j5': 0.57,
                            'right_j6': 3.3161}

    pnp = PickAndPlace(limb, hover_distance)
    # An orientation for gripper fingers to be overhead and parallel to the obj
    overhead_orientation = Quaternion(
                             x=-0.0249590815779,
                             y=0.999649402929,
                             z=0.00737916180073,
                             w=0.00486450832011)
#    overhead_orientation = Quaternion(
#        x=0.7070316149662462,
#        y=0.707181236408181,
#        z=-0.000737845009910686,
#        w=-0.0006707422454052575)

    block_poses = list()
    # The Pose of the block in its initial location.
    # You may wish to replace these poses with estimates
    # from a perception node.
    block_poses.append(Pose(
        position=Point(x=0.7, y=0.15, z=0.020),
        orientation=overhead_orientation))
    # Feel free to add additional desired poses for the object.
    # Each additional pose will get its own pick and place.
    block_poses.append(Pose(
        position=Point(x=0.7, y=-0.15, z=0.020),
        orientation=overhead_orientation))
    # Move to the desired starting angles
    pnp.move_to_start(starting_joint_angles)
    idx = 0
    while not rospy.is_shutdown():
        #print("\nPicking...")
        pnp.pick(block_poses[idx])
        #print("\nPlacing...")
        idx = (idx+1) % len(block_poses)
        pnp.place(block_poses[idx])
    return 0

if __name__ == '__main__':
    sys.exit(main())
