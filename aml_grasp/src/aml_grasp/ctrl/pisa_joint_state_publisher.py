#!/usr/bin/env python


# import roslib; roslib.load_manifest('aml_grasp')
import rospy

from sensor_msgs.msg import JointState

from aml_robot.pisaiit.pisaiit_kinematics import pisaiit_kinematics
from aml_robot.pisaiit.pisaiit_robot import PisaIITHand
from aml_io.io_tools import get_file_path, get_aml_package_path
import PyKDL
import numpy as np

from reach_interface.reach_interface import ReachInterface

from reach_interface.config import default_reach_config

class JointStateMessage():
    def __init__(self, name, position, velocity, effort):
        self.name = name
        self.position = position
        self.velocity = velocity
        self.effort = effort

class JointStatePublisher():
    def __init__(self):
        rospy.init_node('pisa_joint_state_publisher', anonymous=True)
        
        rate = rospy.get_param('~rate', 300)
        r = rospy.Rate(rate)
                                                                
        # Start publisher
        self.joint_states_pub = rospy.Publisher('/joint_states', JointState)
       
        rospy.loginfo("Starting Pisa Joint State Publisher at " + str(rate) + "Hz")


        # self._pisaiit_hand = PisaIITHand()

        # self._models_path = get_aml_package_path('aml_grasp/src/aml_grasp/models')
        # self._hand_path = get_file_path('pisa_hand_right.urdf', models_path)
        # self._hand_kinematics = pisaiit_kinematics(pisaiit_hand,hand_path)

        self._gi = ReachInterface(config = default_reach_config)
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

        names = ["noname"]*33
        velocity = [0.0]*33

        all_angles = little_angles+ring_angles+middle_angles+index_angles+thumb_angles#thumb_angles + index_angles + middle_angles + ring_angles + little_angles

        # Construct message & publish joint states
        msg = JointState()
        msg.name = names
        msg.position = all_angles
        msg.velocity = velocity
        msg.effort = velocity
       
        # for angle in all_angles:
        #     # msg.name.append(joint.name)
        #     msg.position.append(angle)
        #     # msg.velocity.append(joint.velocity)
        #     # msg.effort.append(joint.effort)
           
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'base_link'
        self.joint_states_pub.publish(msg)
        
if __name__ == '__main__':
    try:
        s = JointStatePublisher()
    except rospy.ROSInterruptException: pass