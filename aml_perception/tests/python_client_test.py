#!/usr/bin/env python

import sys
import rospy
from aml_services.srv import PCLUtility
from aml_services.msg import PCLCustomMsg



if __name__ == "__main__":

    print "waiting for service"
    rospy.wait_for_service('aml_pcl_service')
    print "service found"
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


        # try:
        #     msg2 = PCLCustomMsg()
        #     msg2.cloud_1 = resp.msg_out.cloud_1
        #     msg2.float_array_1 = [0., 0., 0., 0., 0., 1.2, -3.2, 0., 0., 3.2, 1.2, 0, 0, 0, 0, 1]
        #     resp2 = client("apply_transformation", msg2)

        #     print resp2.info

            # print resp2.msg_out.cloud_1
            # print resp2.msg_out.float_1

        # try:

        #     msg2 = PCLCustomMsg()
        #     msg2.cloud_1 = resp.msg_out.cloud_1
        #     msg2.float_array_1 = [0.,25., 0., 0., 0., 1.2, -3.2, 0.7, 0., 3.2, 1.2, 0, 0, 0, 0, 1]
        #     resp2 = client("apply_transformation", msg2)

        #     print resp2.info
        
        # try:
        #     # print resp.out_cloud_1
        #     msg2 = PCLCustomMsg()
        #     msg2.cloud_1 = resp.msg_out.cloud_1
        #     msg2.float_array_1 = [0.01,0.01,0.01]
        #     # resp2 = client("downsample_cloud",resp.msg_out.cloud_1,None,None,[0.01,0.01,0.01])
        #     resp2 = client("downsample_cloud",msg2)

        #     print resp2.info

            

        try:
            # print resp.out_cloud_1
            msg2 = PCLCustomMsg()
            msg2.cloud_1 = resp.msg_out.cloud_1
            # resp2 = client("downsample_cloud",resp.msg_out.cloud_1,None,None,[0.01,0.01,0.01])
            resp2 = client("get_points_not_in_plane",msg2)

            print resp2.info

            raw_input()

            try:
                msg3 = PCLCustomMsg()
                msg3.cloud_1 = resp2.msg_out.cloud_1
                msg3.cloud_2 = resp.msg_out.cloud_1
                msg3.string_1 = "/home/saif/Desktop/image_0003.pcd"
                resp3 = client("add_clouds",msg3)

                print len(resp.msg_out.cloud_1.points), "+", len(resp2.msg_out.cloud_1.points)
                print len(resp3.msg_out.cloud_1.points)
                print resp3.info

                try:
                    msg4 = PCLCustomMsg()
                    msg4.cloud_1 = resp3.msg_out.cloud_1
                    msg4.string_1 = "/home/saif/Desktop/image_0003.pcd"
                    resp4 = client("save_to_file",msg4)

                    print resp4.info

                except rospy.ServiceException, e:
                    print "Service call failed: %s"%e

            except rospy.ServiceException, e:
                print "Service call failed: %s"%e

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e


    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


