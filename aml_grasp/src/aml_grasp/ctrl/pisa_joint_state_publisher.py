#!/usr/bin/env python


# import roslib; roslib.load_manifest('aml_grasp')
import rospy

from sensor_msgs.msg import JointState

class JointStateMessage():
    def __init__(self, name, position, velocity, effort):
        self.name = name
        self.position = position
        self.velocity = velocity
        self.effort = effort

class JointStatePublisher():
    def __init__(self):
        rospy.init_node('pisa_joint_state_publisher', anonymous=True)
        
        rate = rospy.get_param('~rate', 20)
        r = rospy.Rate(rate)
                                                                
        # Start publisher
        self.joint_states_pub = rospy.Publisher('/joint_states', JointState)
       
        rospy.loginfo("Starting Pisa Joint State Publisher at " + str(rate) + "Hz")
       
        while not rospy.is_shutdown():
            self.publish_joint_states()
            r.sleep()
           
       
    def publish_joint_states(self):
        # Construct message & publish joint states
        msg = JointState()
        msg.name = []
        msg.position = []
        msg.velocity = []
        msg.effort = []
       
        for joint in self.joint_states.values():
            msg.name.append(joint.name)
            msg.position.append(joint.position)
            msg.velocity.append(joint.velocity)
            msg.effort.append(joint.effort)
           
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'base_link'
        self.joint_states_pub.publish(msg)
        
if __name__ == '__main__':
    try:
        s = JointStatePublisher()
except rospy.ROSInterruptException: pass