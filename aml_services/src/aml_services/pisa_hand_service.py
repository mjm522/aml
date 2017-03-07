#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32
from aml_services.srv import SendPisaHandCmd, ReadPisaHandCurr, SendPisaHandCmdResponse, ReadPisaHandCurrResponse

def send_pos_cmd_pisa_hand(req):
    cmd_pub = rospy.Publisher('soft_hand_pos_cmd', Float32, queue_size=10)
    cmd_pub.publish(req.cmd)
    status = True
    return SendPisaHandCmdResponse(status)

def read_current_pisa_hand(req):
    sh_current_status = rospy.Publisher('soft_hand_read_current', Float32, queue_size=10)
    sh_current_status.publish(3.)
    status = True
    return ReadPisaHandCurrResponse(status)
    
def pisa_hand_service_server():
    rospy.init_node('pisa_hand_service_server')
    pisa_hand_pos_cmd  = rospy.Service('pisa_hand_pos_cmd',  SendPisaHandCmd,  send_pos_cmd_pisa_hand)
    pisa_hand_read_current  = rospy.Service('pisa_hand_read_current',  ReadPisaHandCurr,  read_current_pisa_hand)
    rospy.spin()

if __name__ == "__main__":
    pisa_hand_service_server()