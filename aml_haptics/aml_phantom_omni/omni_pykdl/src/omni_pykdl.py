#!/usr/bin/env python 

import rospy
import math
import tf
from tf.transformations import euler_from_quaternion
from tf.transformations import compose_matrix 
from tf.transformations import is_same_transform

from sensor_msgs.msg import JointState
from urdf_parser_py.urdf import URDF

import numpy as np
import PyKDL as kdl
from aml_pykdl import AMLPyKDL
from kdl_utils import kdl_tree_from_urdf_model

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
    

    #tf_base_to_stylus uses tf to look up the transfrom from the frame located at the root of the Omni's URDF (/base) to the frame
    #at the end of the URDF (/stylus).
    # def tf_base_to_stylus (self):

    #     try:
    #         (self.transl, self.quat) = self._tf_listener.lookupTransform('base', 'tip', rospy.Time(0))
    #         #lookupTransrom is a method which returns the transfrom between two coordinate frames. 
    #         #lookupTransfrom returns a translation and a quaternion
    #         self.rot= euler_from_quaternion(self.quat) #Get euler angles from the quaternion

    #         self.tf_SE3 = compose_matrix(angles=self.rot,translate=self.transl)

    #         #Store the transformation in a format compatible with gemoetr_msgs/Twist
    #         self.transf = Twist(Vector3(self.transl[0],self.transl[1],self.transl[2]),(Vector3(self.rot[0],self.rot[1],self.rot[2])))
    #         #Publish the transformation
    #         self.Tsb.publish(self.transf)

    #     except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
    #         self.tf_SE3 = None
    #         pass
    #         #If exceptions occur, skip trying to lookup the transform. 
    #         #It is encouraged to publish a logging message to rosout with rospy. 
    #         #i.e: 
    #         #rospy.logerr("Could not transform from %s to %s,"base","stylus")


    # def kdl_kinematics (self, data):

    #     self.q_sensors = data
    #     self.tree = kdl.kdl_tree_from_urdf_model(self._omni_urdf) # create a kdl tree from omni URDF model
    #     self._omni_urdf_kin = kdl.KDLKinematics(self._omni_urdf, "base", "tip") # create instance that captures the kinematics of the robot arm  

    #     #Forward Kinematics
    #     self.pose_stylus = self._omni_urdf_kin.forward(data) #compute the forward kinematics from the sensor joint angle position using the kinematics from the kdl tree


    #     #Inverse Kinematics
    #     self.q_guess = np.array(data)+0.2 #make an initial guess for your joint angles by deviating all the sensor joint angles by 0.2
    #     self.q_ik = self._omni_urdf_kin.inverse(self.pose_stylus, self.q_guess) #using the position from the forward kinematics 'pose_stylus' and the offset initial guess, compute 
    #     #the desired joint angles for that position.

    #     self.delta_q = self.q_ik-data


    # def print_output (self,data):

    #     if (self.tf_SE3 is not None) and (self.q_sensors is not None):
    #         print "Base to stylus using tf:","\n", self.tf_SE3, "\n"
    #         print "base to stylus using KDL forward kinematics", "\n" , self.pose_stylus, "\n"
    #         print "transformations.is.same_transform tf vs KDL forward kin:", "\n", is_same_transform(self.tf_SE3,self.pose_stylus), "\n"
    #         print "Joint angles from sensors q_sensors:","\n" , self.q_sensors, "\n"
    #         print "Joint state angles from KDL inverse kinematics q_ik", "\n" , self.q_ik , "\n"
    #         print "The difference delta_q between q_sensors and q_ik", "\n" , self.delta_q , "\n\n\n\n\n\n\n\n\n"


def main():
    """
    Run the main loop, by instatiating a System class, and then
    calling ros.spin
    """
    rospy.init_node('omni_pykdl_node')

    opl = OmniPyKDL()

    rospy.spin()

if __name__=='__main__':
    main()
