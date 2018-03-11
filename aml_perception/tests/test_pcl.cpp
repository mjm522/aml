#include "ros/ros.h"
#include "aml_services/PCLUtility.h"
#include <cstdlib>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pcl_test_client");
  // if (argc != 2)
  // {
  //   ROS_INFO("usage: clnt string");
  //   return 1;
  // }

  ros::NodeHandle n;
  ros::ServiceClient client = n.serviceClient<aml_services::PCLUtility>("aml_pcl_service");
  aml_services::PCLUtility srv;
  srv.request.function = "read_pcd_file";
  // srv.request.b = (argv[2]);
  srv.request.msg_in.string_1 = "/home/saif/Desktop/image_0001.pcd";



  if (client.call(srv))
  {
    ROS_INFO("%s: %s", srv.request.function.c_str(), srv.response.info.c_str());

    aml_services::PCLUtility srv2;
    srv2.request.function = "save_to_file";
    // srv.request.b = (argv[2]);
    srv2.request.msg_in.string_1 = "/home/saif/Desktop/image_0002.pcd";
    srv2.request.msg_in.cloud_1 = srv.response.msg_out.cloud_1;

    if (client.call(srv2))
    {
        ROS_INFO("%s: %s", srv2.request.function.c_str(), srv2.response.info.c_str());   
    }
  }
  else
  {
    ROS_ERROR("Failed to call service aml_pcl_service");
    return 1;
  }



  return 0;
}