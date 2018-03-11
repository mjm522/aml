#!/usr/bin/env python

import sys
import rospy
from aml_services.srv import PCLUtility
from aml_perception.msg import PCLCustomMsg

'''
    PCLCustomMsg.msg
        sensor_msgs/PointCloud cloud_1
        sensor_msgs/PointCloud cloud_2
        string string_1
        float32[] float_array_1

    PCLUtility.srv
        string function
        aml_perception/PCLCustomMsg msg_in
        ---
        aml_perception/PCLCustomMsg msg_out
        string info
        bool success

'''



if __name__ == "__main__":


    rospy.wait_for_service('aml_pcl_service')
    try:
        client = rospy.ServiceProxy('aml_pcl_service', PCLUtility)

        # request_msg.in_cloud_1 = None
        # request_msg.in_cloud_2 = None
        msg1 = PCLCustomMsg()
        msg1.string_1 = "/home/saif/Desktop/image_0001.pcd"
        # resp = client("read_pcd_file",None,None,"/home/saif/Desktop/image_0001.pcd",None)
        resp = client("read_pcd_file",msg1)

        print resp.info

        raw_input()
        
        try:
            # print resp.out_cloud_1
            msg2 = PCLCustomMsg()
            msg2.cloud_1 = resp.msg_out.cloud_1
            msg2.float_array_1 = [0.01,0.01,0.01]
            # resp2 = client("downsample_cloud",resp.msg_out.cloud_1,None,None,[0.01,0.01,0.01])
            resp2 = client("downsample_cloud",msg2)

            print resp2.info

            raw_input()

            try:
                msg3 = PCLCustomMsg()
                msg3.cloud_1 = resp2.msg_out.cloud_1
                msg3.string_1 = "/home/saif/Desktop/image_0003.pcd"
                resp3 = client("save_to_file",msg3)

                print resp3.info

            except rospy.ServiceException, e:
                print "Service call failed: %s"%e

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e


    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


