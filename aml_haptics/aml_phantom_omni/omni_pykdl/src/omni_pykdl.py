#!/usr/bin/env python 

import rospy
import numpy as np
import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from sensor_msgs.msg import JointState
from aml_utils.aml_pykdl.aml_pykdl import AMLPyKDL
from aml_utils.aml_pykdl.pykdl_utils import kdl_tree_from_urdf_model
from geometry_msgs.msg import Twist, Vector3, Pose, Point, Quaternion


class OmniPyKDL(AMLPyKDL):

    def __init__(self):

        self._pose_pub = rospy.Publisher('omni1_ee_state', Pose)

        self._omni_urdf =  URDF.from_parameter_server() #URDF.from_xml_file('../../omni_description/urdf/omni.urdf') #

        self._js_sub = rospy.Subscriber("omni1_joint_states", JointState, self.get_joint_states)

        chain = self.setup_chain()

        self._base_position = np.zeros(3)

        self._curr_js = None

        AMLPyKDL.__init__(self, chain, self._base_position)

    def setup_chain(self):

        self._kdl_tree = kdl_tree_from_urdf_model(self._omni_urdf)

        return self._kdl_tree.getChain('base', 'tip')

    def get_ee_pose(self, q = None):
        
        if q is None:
            
            if self._q is None:
                
                return
            
            else:

                q = self._curr_js

        self.update_chain(q=q)

        ee_pos, ee_ori = self.compute_fwd_kinematics(ori_type='quat')

        pos = Point(ee_pos[0],ee_pos[1], ee_pos[2])

        ori = Quaternion(ee_ori[1], ee_ori[2], ee_ori[3] ,ee_ori[0])

        pose_msg = Pose( pos, ori )

        self._pose_pub.publish(pose_msg)


    def get_joint_states(self, data):
            
        self._curr_js = data.position

        self.get_ee_pose()
    

def main():

    rospy.init_node('omni_pykdl_node')

    opl = OmniPyKDL()

    rospy.spin()

if __name__=='__main__':
    main()
