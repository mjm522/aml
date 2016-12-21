import rospy
from aml_robot.baxter_robot import BaxterArm

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

import struct

class IKBaxter():

    def __init__(self, limb='right'):
        self._limb = limb
        self.iksvc = None
        self.ikreq = None

        self.arm = BaxterArm(self._limb)

        self.configure_ik_service()

    def configure_ik_service(self):
        self.ns = "ExternalTools/" + self._limb + "/PositionKinematicsNode/IKService"
        self.iksvc = rospy.ServiceProxy(self.ns, SolvePositionIK)
        self.ikreq = SolvePositionIKRequest()

    def test_pose(self):

        #these are two valid posses which should
        #generate valid solutions, kept here for 
        #debugging purposes, takes from ik_service_client.py in baxter_examples

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

    def ik_servive_request(self, pos, ori):
        #remember the ori will be in w, x,y,z format as opposed to usual ROS format
        success_flag = False
        #incase of failure return back current joints
        limb_joints  = self.arm._state['position'] 

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
        try:
            rospy.wait_for_service(self.ns, 5.0)
            resp = self.iksvc(self.ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))

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
            print "\nIK Joint Solution:\n", limb_joints

            success_flag = True
        else:
            print("INVALID POSE - No Valid Joint Solution Found.")

        return limb_joints

if __name__ == "__main__":
    rospy.init_node('poke_box', anonymous=True)
    ik_baxter = IKBaxter(limb='right')
    pos, ori = ik_baxter.test_pose()
    ik_baxter.ik_servive_request(pos=pos, ori=ori)