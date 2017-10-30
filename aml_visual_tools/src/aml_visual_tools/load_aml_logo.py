import os
import cv2
import rospy
import cv_bridge
from sensor_msgs.msg import Image
from os.path import dirname, abspath

def load_aml_logo(display_topic = '/robot/xdisplay'):
    """
    Send the image located at the specified path to the head
    display on Baxter.

    @param path: path to the image file to load and send
    """
    logo_file = '/'.join(dirname(dirname(abspath(__file__))).split('/')[:-2]) + '/aml_assets/aml_logo/aml_logo.jpg'

    if not os.access(logo_file, os.R_OK):
        rospy.loginfo("Cannot read file at '%s'" % (logo_file))
        return 

    img = cv2.imread(logo_file)
    msg = cv_bridge.CvBridge().cv2_to_imgmsg(img, encoding="bgr8")
    pub = rospy.Publisher(display_topic, Image, latch=True, queue_size=1)
    pub.publish(msg)
    # Sleep to allow for image to be published.
    rospy.sleep(2)
