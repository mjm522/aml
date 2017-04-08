import rospy
from sensor_msgs.msg import Image
from aml_visual_tools.visual_tools import show_image

scene_image = None

def perception_callback(image_data):
    global scene_image
    scene_image = image_data

def main(time_out=5.):

    global scene_image

    rospy.init_node('test_camera_sensor', anonymous=True)
    rospy.Subscriber('/rgb_image_out', Image, perception_callback)

    start_time = rospy.Time.now()
    timeout = rospy.Duration(time_out) # Timeout of 'time_out' seconds
    while scene_image is None:

        if (rospy.Time.now() - start_time > timeout):
            raise Exception("Time out reached, image is None")
        else:
            continue
    show_image(scene_image)

if __name__ == '__main__':
    main()