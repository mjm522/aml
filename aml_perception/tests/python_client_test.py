#!/usr/bin/env python

import sys
import rospy
from aml_services.srv import PCLUtility



if __name__ == "__main__":


    rospy.wait_for_service('aml_pcl_service')
    try:
        client = rospy.ServiceProxy('aml_pcl_service', PCLUtility)

        # request_msg.in_cloud_1 = None
        # request_msg.in_cloud_2 = None
        resp = client("read_pcd_file",None,None,"/home/saif/Desktop/image_0001.pcd",None)

        print resp.info

        raw_input()
        
        try:
            # print resp.out_cloud_1
            resp2 = client("downsample_cloud",resp.out_cloud_1,None,None,[0.01,0.01,0.01])

            print resp2.info

            raw_input()

            try:
                resp3 = client("save_to_file",resp2.out_cloud_1,None,"/home/saif/Desktop/image_0003.pcd",None)

                print resp3.info

            except rospy.ServiceException, e:
                print "Service call failed: %s"%e

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e


    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


