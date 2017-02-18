import os
import sys
import argparse
 
import rospy
import xacro_jade

from os.path import dirname, abspath
from baxter_core_msgs.msg import URDFConfiguration

def xacro_parse(filename):
    doc = xacro_jade.parse(None, filename)
    xacro_jade.process_doc(doc, in_order=True)
    return doc.toprettyxml(indent='  ')


def send_urdf(parent_link, root_joint, urdf_filename):
    """
   Send the URDF Fragment located at the specified path to
   modify the electric gripper on Baxter.
   @param parent_link: parent link to attach the URDF fragment to
                       (usually <side>_hand)
   @param root_joint: root link of the URDF fragment (usually <side>_gripper_base)
   @param urdf_filename: path to the urdf XML file to load into xacro and send
   """
    msg = URDFConfiguration()
    # The updating the time parameter tells
    # the robot that this is a new configuration.
    # Only update the time when an updated internal
    # model is required. Do not continuously update
    # the time parameter.
    msg.time = rospy.Time.now()
    # link to attach this urdf to onboard the robot
    msg.link = parent_link
    # root linkage in your URDF Fragment
    msg.joint = root_joint
    msg.urdf = xacro_parse(urdf_filename)
    pub = rospy.Publisher('/robot/urdf', URDFConfiguration, queue_size=10)
    rate = rospy.Rate(5) # 5hz
    while not rospy.is_shutdown():
        # Only one publish is necessary, but here we
        # will continue to publish until ctrl+c is invoked
        pub.publish(msg)
        rate.sleep()

def main():

    urdf_folder_path = dirname(dirname(abspath(__file__))) + '/urdf_addendums/'

    """RSDK URDF Fragment Example:
   This example shows a proof of concept for
   adding your URDF fragment to the robot's
   onboard URDF (which is currently in use).
   """
    # arg_fmt = argparse.RawDescriptionHelpFormatter
    # parser = argparse.ArgumentParser(formatter_class=arg_fmt,
    #                                  description=main.__doc__)
    # required = parser.add_argument_group('required arguments')
    # required.add_argument(
    #     '-f', '--file', metavar='PATH', required=True,
    #     help='Path to URDF file to send'
    # )
    # required.add_argument(
    #     '-l', '--link', required=False, default="left_hand",
    #     help='URDF Link already to attach fragment to (usually <left/right>_hand)'
    # )
    # required.add_argument(
    #     '-j', '--joint', required=False, default="left_gripper_base",
    #     help='Root joint for fragment (usually <left/right>_gripper_base)'
    # )
    # args = parser.parse_args(rospy.myargv()[1:])

    link_name  = "left_hand"
    joint_name = "left_gripper_base"
    file_name  = urdf_folder_path + "left_end_effector.urdf.xacro"

    print file_name
 
    rospy.init_node('rsdk_configure_urdf', anonymous=True)
 
    # if not os.access(args.file, os.R_OK):
    #     rospy.logerr("Cannot read file at '%s'" % (args.file,))
    #     return 1
    # send_urdf(args.link, args.joint, args.file)
    send_urdf(link_name, joint_name, file_name)
    return 0

if __name__ == '__main__':
    sys.exit(main())