#!/usr/bin/env python


import roslib; roslib.load_manifest('aml_grasp')
import rospy
import tf

from sensor_msgs.msg import JointState

from aml_robot.pisaiit.pisaiit_kinematics import pisaiit_kinematics
from aml_robot.pisaiit.pisaiit_robot import PisaIITHand
from aml_io.io_tools import get_file_path, get_aml_package_path
import PyKDL
import numpy as np

from reach_interface.reach_interface import ReachInterface

from reach_interface.config import default_reach_config

link_dict ={
        'thumb_links': ['soft_hand_kuka_coupler_bottom',
                        'soft_hand_kuka_coupler',
                        'soft_hand_clamp',
                        'soft_hand_softhand_base',
                        'soft_hand_palm_link',
                        'soft_hand_thumb_knuckle_link',
                        'soft_hand_thumb_fake_link1',
                        'soft_hand_thumb_proximal_link'
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
        
        rate = rospy.get_param('~rate', 300)
        r = rospy.Rate(rate)
                                                                
        # Start publisher
        self.joint_states_pub = rospy.Publisher('/joint_states', JointState)
       
        rospy.loginfo("Starting Pisa Joint State Publisher at " + str(rate) + "Hz")


        self._pisaiit_hand = PisaIITHand()

        self._models_path = get_aml_package_path('aml_grasp/src/aml_grasp/models')
        self._hand_path = get_file_path('pisa_hand_right.urdf', self._models_path)
        self._hand_kinematics = pisaiit_kinematics(self._pisaiit_hand, self._hand_path)

        self._gi = ReachInterface(config = default_reach_config)


        self._br = tf.TransformBroadcaster()



        while not rospy.is_shutdown():
            self.publish_joint_states()
            r.sleep()



           
       
    def publish_joint_states(self):


        # Read glove state

        
        self._gi.update()
        flex_state = self._gi.get_flex_state()

        thumb_angles = [flex_state[0]]*5
        
        index_angles = [flex_state[1]]*7
        index_angles[0] = 0.0
        
        middle_angles = [flex_state[2]]*7
        middle_angles[0] = 0.0
        
        ring_angles = [flex_state[3]]*7
        ring_angles[0] = 0.0
        
        little_angles = [flex_state[4]]*7
        little_angles[0] = 0.0

        names = ["hand_joint%d"%(joint,) for joint in range(33)]
        velocity = [0.0]*33

        finger_list = [thumb_angles,index_angles,middle_angles,ring_angles,little_angles]

        all_angles = thumb_angles + index_angles + middle_angles + ring_angles + little_angles

        # Construct message & publish joint states
        msg = JointState()
        msg.name = names
        msg.position = all_angles
        msg.velocity = velocity
        msg.effort = velocity
           
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'right_hand'


        # compute forward kinematics with current angles
        poses = []

        for i in range(len(finger_list)):
            poses.append(dict(zip(link_dict[link_dict['finger_links'][i]],
                             self._hand_kinematics.forward_position_kinematics(finger_list[i],i))))

        self.joint_states_pub.publish(msg)

        for pose in poses:
            for k, p in pose.items():

                self._br.sendTransform(tuple(p[:3]),
                                 tuple(p[3:]),
                                 rospy.Time.now(),
                                 k,"right_hand")


        
        
if __name__ == '__main__':
    try:
        s = JointStatePublisher()
    except rospy.ROSInterruptException: 
        pass