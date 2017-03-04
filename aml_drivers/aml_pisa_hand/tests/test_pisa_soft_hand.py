import rospy
from std_msgs.msg import Float32

def send_pos_cmd():
    
    rospy.init_node('send_pos_cmd', anonymous=True)

    cmd_pub = rospy.Publisher('soft_hand_pos_cmd', Float32, queue_size=10)
    sh_current_status = rospy.Publisher('soft_hand_read_current', Float32, queue_size=10)
    
    rate = rospy.Rate(10) # 10hz

    while not rospy.is_shutdown():

        cmd = raw_input('Enter position command (between 0 and 1 to open, 3 to read status)')

        if float(cmd) > 2.:
            sh_current_status.publish(int(cmd))
        else:
            cmd_pub.publish(float(cmd))

        rate.sleep()

if __name__ == '__main__':
    try:
        send_pos_cmd()
    except rospy.ROSInterruptException:
        pass