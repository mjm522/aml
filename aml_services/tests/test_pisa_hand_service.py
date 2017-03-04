import rospy
import numpy as np
from aml_services.srv import SendPisaHandCmd, ReadPisaHandCurr


def pisa_hand_service_send_pos_client(cmd):
    rospy.wait_for_service('pisa_hand_pos_cmd')
    try:
        pisa_hand_service_ = rospy.ServiceProxy('pisa_hand_pos_cmd', SendPisaHandCmd)
        pisa_hand_service_(cmd)
    except rospy.ServiceException, e:
        print "Service call to pisa_hand_pos_cmd failed: %s"%e

def pisa_hand_service_read_curr_client(cmd):
    rospy.wait_for_service('pisa_hand_read_current')
    try:
        pisa_hand_service_ = rospy.ServiceProxy('pisa_hand_read_current', ReadPisaHandCurr)
        pisa_hand_service_(cmd)
    except rospy.ServiceException, e:
        print "Service call to pisa_hand_read_current failed: %s"%e


def main():
    print "Test of pisa hand push service"
    cmd = raw_input('Enter position command (between 0 and 1 to close)')
    status = pisa_hand_service_send_pos_client(float(cmd))
    print "Response from the service \t", status

if __name__ == '__main__':
    main()