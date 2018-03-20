import rospy
import numpy as np

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

from sensor_msgs.msg import JointState

import struct


class IKBaxter(object):
    def __init__(self, limb):

        self._limb = limb.name
        self.iksvc = None
        self.ikreq = None

        self.arm = limb

        self.configure_ik_service()

    def configure_ik_service(self):
        self.ns = "ExternalTools/" + self._limb + "/PositionKinematicsNode/IKService"
        self.iksvc = rospy.ServiceProxy(self.ns, SolvePositionIK)

    def test_pose(self):

        # these are two valid posses which should
        # generate valid solutions, kept here for
        # debugging purposes, takes from ik_service_client.py in baxter_examples

        # pos in x,y,z format
        # ori in w,x,y,z format

        left_pose = {}
        left_pose['pos'] = [0.657579481614, 0.851981417433, 0.0388352386502]
        left_pose['ori'] = [0.262162481772, -0.366894936773, 0.885980397775, 0.108155782462]

        right_pose = {}
        right_pose['pos'] = [0.656982770038, -0.852598021641, 0.0388609422173]
        right_pose['ori'] = [0.261868353356, 0.367048116303, 0.885911751787, -0.108908281936]

        if self._limb == 'left':
            return left_pose['pos'], left_pose['ori']
        else:
            return right_pose['pos'], right_pose['ori']

    def ik_service_request(self, pos, ori, seed_angles=None, null_space_goal=None, use_advanced_options=False):

        self.ikreq = SolvePositionIKRequest()

        # remember the ori will be in w, x,y,z format as opposed to usual ROS format
        success_flag = False
        # incase of failure return back current joints
        limb_joints = self.arm._state['position']

        resp = None

        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ik_msg = PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(
                    x=pos[0],
                    y=pos[1],
                    z=pos[2],
                ),
                orientation=Quaternion(
                    x=ori[1],
                    y=ori[2],
                    z=ori[3],
                    w=ori[0],
                ),
            )
        )
        self.ikreq.pose_stamp.append(ik_msg)

        if use_advanced_options:
            # Optional Advanced IK parameters
            # rospy.loginfo("Running Advanced IK Service Client example.")
            # The joint seed is where the IK position solver starts its optimization
            self.ikreq.seed_mode = self.ikreq.SEED_USER
            seed = JointState()
            seed.name = self.arm.joint_names()

            if seed_angles is None:
                seed.position = [0.7, 0.4, -1.7, 1.4, -1.1, -1.6, -0.4]
            else:
                seed.position = seed_angles
            self.ikreq.seed_angles.append(seed)

            # Once the primary IK task is solved, the solver will then try to bias the
            # the joint angles toward the goal joint configuration. The null space is
            # the extra degrees of freedom the joints can move without affecting the
            # primary IK task.

            if null_space_goal is not None:
                self.ikreq.use_nullspace_goal.append(True)
                # The nullspace goal can either be the full set or subset of joint angles
                goal = JointState()
                goal.name = self.arm.joint_names()
                
                goal.position = null_space_goal
                self.ikreq.nullspace_goal.append(goal)
                # The gain used to bias toward the nullspace goal. Must be [0.0, 1.0]
                # If empty, the default gain of 0.4 will be used
                self.ikreq.nullspace_gain.append(0.4)

        else:
            pass
            # rospy.loginfo("Running Simple IK Service Client example.")

        try:
            rospy.wait_for_service(self.ns, 5.0)
            resp = self.iksvc(self.ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))

        try:

            # Check if result valid, and type of seed ultimately used to get solution
            # convert rospy's string representation of uint8[]'s to int's
            resp_seeds = struct.unpack('<%dB' % len(resp.result_type),
                                       resp.result_type)
            if (resp_seeds[0] != resp.RESULT_INVALID):
                seed_str = {
                    self.ikreq.SEED_USER: 'User Provided Seed',
                    self.ikreq.SEED_CURRENT: 'Current Joint Angles',
                    self.ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                }.get(resp_seeds[0], 'None')
                print("SUCCESS - Valid Joint Solution Found from Seed Type: %s" %
                      (seed_str,))
                # Format solution into Limb API-compatible dictionary
                limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
                # print "\nIK Joint Solution:\n", limb_joints

                success_flag = True

                joint_names = self.arm.joint_names()

                def to_list(ls):
                    return [ls[n] for n in joint_names]

                limb_joints = np.array(to_list(limb_joints))

            else:
                print("INVALID POSE - No Valid Joint Solution Found.")
        except:
            print("INVALID POSE - No Valid Joint Solution Found.")

        return success_flag, limb_joints
