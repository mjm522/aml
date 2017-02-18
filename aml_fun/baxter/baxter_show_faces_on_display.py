import os
import sys
import argparse

import rospy

import cv2
import cv_bridge

from sensor_msgs.msg import (
    Image,
)

from os.path import dirname, abspath

faces_folder_path = '/'.join(dirname(dirname(abspath(__file__))).split('/')[:-1]) + '/aml_assets/baxter_faces/'

def send_image(path):
    """
    Send the image located at the specified path to the head
    display on Baxter.

    @param path: path to the image file to load and send
    """
    img = cv2.imread(path)
    msg = cv_bridge.CvBridge().cv2_to_imgmsg(img, encoding="bgr8")
    pub = rospy.Publisher('/robot/xdisplay', Image, latch=True, queue_size=1)
    pub.publish(msg)
    # Sleep to allow for image to be published.
    rospy.sleep(2)

def get_face_names():
    face_names = [];
    for file in os.listdir(faces_folder_path):
        face_names.append(file)

    return face_names

def main():

    rospy.init_node('rsdk_xdisplay_image', anonymous=True)

    face_names = get_face_names()

    # file = faces_folder_path + 'SillyNEWhite.jpg'

    # while not rospy.is_shutdown():
    #     send_image(file)

    for face in face_names:

        file = faces_folder_path + face

        if not os.access(file, os.R_OK):
            rospy.logerr("Cannot read file at '%s'" % (args.file,))
            return 1

        send_image(file)
        

if __name__ == '__main__':
    sys.exit(main())