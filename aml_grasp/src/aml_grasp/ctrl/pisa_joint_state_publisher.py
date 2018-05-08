#!/usr/bin/env python


import roslib; roslib.load_manifest('aml_grasp')
import rospy
import tf
import tf2_ros

from sensor_msgs.msg import JointState

from aml_robot.pisaiit.pisaiit_kinematics import pisaiit_kinematics
from aml_robot.pisaiit.pisaiit_robot import PisaIITHand
from aml_io.io_tools import get_file_path, get_aml_package_path
import PyKDL
import numpy as np

from reach_interface.reach_interface import ReachInterface

from reach_interface.config import default_reach_config

link_dict ={
        'palm_links' :['soft_hand_kuka_coupler_bottom',
                        'soft_hand_kuka_coupler',
                        'soft_hand_clamp',
                        'soft_hand_softhand_base',
                        'soft_hand_palm_link'],
        'thumb_links': ['soft_hand_kuka_coupler_bottom',
                        'soft_hand_kuka_coupler',
                        'soft_hand_clamp',
                        'soft_hand_softhand_base',
                        'soft_hand_palm_link',
                        'soft_hand_thumb_knuckle_link',
                        'soft_hand_thumb_fake_link1',
                        'soft_hand_thumb_proximal_link',
                        'soft_hand_thumb_fake_link3',
                        'soft_hand_thumb_distal_link'],
         'index_links': ['soft_hand_kuka_coupler_bottom',
                        'soft_hand_kuka_coupler',
                        'soft_hand_clamp',
                        'soft_hand_softhand_base',
                        'soft_hand_palm_link',
                        'soft_hand_index_knuckle_link',
                        'soft_hand_index_fake_link1',
                        'soft_hand_index_proximal_link',
                         'soft_hand_index_fake_link2',
                         'soft_hand_index_middle_link',
                         'soft_hand_index_fake_link3',
                         'soft_hand_index_distal_link'],
        'middle_links': ['soft_hand_kuka_coupler_bottom',
                        'soft_hand_kuka_coupler',
                        'soft_hand_clamp',
                        'soft_hand_softhand_base',
                        'soft_hand_palm_link',
                        'soft_hand_middle_knuckle_link',
                         'soft_hand_middle_fake_link1',
                         'soft_hand_middle_proximal_link',
                         'soft_hand_middle_fake_link2',
                         'soft_hand_middle_middle_link',
                         'soft_hand_middle_fake_link3',
                         'soft_hand_middle_distal_link'],
        'ring_links':   ['soft_hand_kuka_coupler_bottom',
                        'soft_hand_kuka_coupler',
                        'soft_hand_clamp',
                        'soft_hand_softhand_base',
                        'soft_hand_palm_link',
                        'soft_hand_ring_knuckle_link',
                         'soft_hand_ring_fake_link1',
                         'soft_hand_ring_proximal_link',
                         'soft_hand_ring_fake_link2',
                         'soft_hand_ring_middle_link',
                         'soft_hand_ring_fake_link3',
                         'soft_hand_ring_distal_link'],
        'little_links': ['soft_hand_kuka_coupler_bottom',
                        'soft_hand_kuka_coupler',
                        'soft_hand_clamp',
                        'soft_hand_softhand_base',
                        'soft_hand_palm_link',
                        'soft_hand_little_knuckle_link',
                         'soft_hand_little_fake_link1',
                         'soft_hand_little_proximal_link',
                         'soft_hand_little_fake_link2',
                         'soft_hand_little_middle_link',
                         'soft_hand_little_fake_link3',
                         'soft_hand_little_distal_link'],
        'finger_links': ['thumb_links', 'index_links','middle_links','ring_links','little_links']
        }


class JointStatePublisher():
    def __init__(self):
        rospy.init_node('pisa_joint_state_publisher', anonymous=True)
        
        rate = rospy.get_param('~rate', 500)
        r = rospy.Rate(rate)
                                                                
        # Start publisher
        self.joint_states_pub = rospy.Publisher('/pisaiit/joint_states', JointState,queue_size=1)
       
        rospy.loginfo("Starting Pisa Joint State Publisher at " + str(rate) + "Hz")


        self._pisaiit_hand = PisaIITHand()

        self._models_path = get_aml_package_path('aml_grasp/src/aml_grasp/models')
        self._hand_path = get_file_path('pisa_hand_right_nomass.urdf', self._models_path)
        self._hand_kinematics = pisaiit_kinematics(self._pisaiit_hand, self._hand_path)

        self._gi = ReachInterface(config=default_reach_config)


        self._br = tf.TransformBroadcaster()

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self._dim = np.array([5,7,7,7,7])
        self._n_joints = np.sum(self._dim)
        self._joint_names = ["hand_joint%d" % (joint,) for joint in range(self._n_joints)]

        self._finger_ids = []
        init = 0
        for d in self._dim:
            self._finger_ids.append(range(init, init+d))
            init += d

        self._all_angles = np.zeros(self._n_joints)
        self._velocity = np.zeros(self._n_joints)

        while not rospy.is_shutdown():
            self.publish_joint_states()
            r.sleep()


    def get_transform(self, frame_name, base_frame):

        # return (0,0,0), (0,0,0,1), rospy.Time.now()

        # t = (0,0,0)
        # q = (0,0,0,1)
        t = None
        try:
            t = self._tf_buffer.lookup_transform(frame_name, base_frame, rospy.Time(), rospy.Duration(5.0))

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Transform could not be queried.")

            return (0,0,0), (0,0,0,1), rospy.Time.now()


        translation = (t.transform.translation.x,t.transform.translation.y,t.transform.translation.z)
        quaternion = (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w)

        return translation, quaternion, t.header.stamp


    def to_transform_matrix(self,t,q):
        t_mat = tf.transformations.translation_matrix(t)
        r_mat = tf.transformations.quaternion_matrix(q)
        transform_mat = np.dot(t_mat, r_mat)

        return transform_mat


    def multiply_transforms(self,t1,q1,t2,q2):

        transform_mat1 = self.to_transform_matrix(t1,q1)
        transform_mat2 = self.to_transform_matrix(t2, q2)

        transform_mat3 = np.dot(transform_mat1, transform_mat2)

        t = tf.transformations.translation_from_matrix(transform_mat3)
        q = tf.transformations.quaternion_from_matrix(transform_mat3)

        return t, q


       
    def publish_joint_states(self):


        # Read glove state

        
        self._gi.update()
        flex_state = self._gi.get_flex_state()


        for fid in range(5):
            self._all_angles[self._finger_ids[fid]] = flex_state[fid]

            if fid > 0:
                self._all_angles[self._finger_ids[fid][0]] = 0.0






        # Construct message & publish joint states
        msg = JointState()
        msg.name = self._joint_names
        msg.position = self._all_angles
        msg.velocity = self._velocity
        msg.effort = self._velocity
           

        msg.header.frame_id = 'right_hand'


        # compute forward kinematics with current angles
        poses = []

        for i in range(5):
            finger_angles = self._all_angles[self._finger_ids[i]]
            finger_link_names = link_dict[link_dict['finger_links'][i]]
            finger_link_poses = self._hand_kinematics.forward_position_kinematics(finger_angles, i)
            # print finger_link_names
            poses.append(dict(zip(finger_link_names, finger_link_poses)))




        right_hand_t, right_hand_q, stamp = self.get_transform('base','right_hand')
        # print right_hand_t, right_hand_q, stamp
        now = stamp
        for pose in poses:
            for k, p in pose.items():
                p_out, q_out = self.multiply_transforms(right_hand_t, right_hand_q,
                                                            tuple(p[:3]), tuple(p[3:]))
                self._br.sendTransform(p_out,
                                       q_out,
                                       now,
                                       k,"base")

        msg.header.stamp = now
        self.joint_states_pub.publish(msg)
        
        
if __name__ == '__main__':
    try:
        s = JointStatePublisher()
    except rospy.ROSInterruptException: 
        pass