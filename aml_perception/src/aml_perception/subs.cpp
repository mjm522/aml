#include "pcl_processing.h"
#include <ros/ros.h>

int main (int argc, char** argv)
{


  std::string file = "/home/saif/Desktop/image_0001.pcd";


  aml_pcl::PCLProcessor processor;

  aml_pcl::PointCloudPtr cloud = processor.getCloudFromPcdFile(file);


  // sensor_msgs::PointCloud2 ptcld;

  ros::init (argc, argv, "pub_pcl");
  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2> ("points2", 1);
  // std::cout << "Loaded "
  //           << cloud->width * cloud->height
  //           << " data points from test_pcd.pcd with the following fields: "
  //           << std::endl;
  // for (size_t i = 0; i < cloud->points.size (); ++i)
  //   std::cout << "    " << cloud->points[i].x
  //             << " "    << cloud->points[i].y
  //             << " "    << cloud->points[i].z << std::endl;

  // return (0);

  sensor_msgs::PointCloud2::Ptr msg = processor.pcl_ros_converter_->ROSMsgFromPclCloud(*cloud);


  aml_pcl::PointCloudPtr cloud2 = processor.getPointsNotInPlane(cloud);

  std::cout << cloud2->size() << std::endl;

  // std::cout << cloud==cloud2 << std::endl;




  // std::cout <<"here "<< msg->height << std::endl;
  // // // msg->header.frame_id = "some_tf_frame";
  // // // // msg->height = msg->width = 1;
  // ros::Rate loop_rate(5);
  //   std::cout << "starting" << std::endl;
  //   while (nh.ok())
  //   {
  //   // msg->header.stamp = ros::Time::now().toNSec();
  //   pub.publish(msg);
  //   ros::spinOnce ();
  //   loop_rate.sleep ();
  //   }
  // std::cout <<"finish" << std::endl;
}

// #include <boost/foreach.hpp>


// void callback(const aml_pcl::PointCloud::ConstPtr& msg)
// {
//   // printf ("Cloud: width = %d, height = %d\n", msg->width, msg->height);
//   // BOOST_FOREACH (const aml_pcl::CloudPoint &pt, msg->points)
//     sensor_msgs::PointCloud2 sm_pc2;
//     sensor_msgs::convertPointCloudToPointCloud2(*msg, sm_pc2)
//     processor.pcl_ros_converter_.pclCloudRGBFromROSMsg(sm_pc2)
//     // printf ("\t(%f, %f, %f)\n", pt.x, pt.y, pt.z);
// }

// int main(int argc, char** argv)
// {

//     aml_pcl::ProcessPCL processor;

//     ros::init(argc, argv, "sub_pcl");
//     ros::NodeHandle nh;
//     ros::Subscriber sub = nh.subscribe<aml_pcl::PointCloud>("points2", 1, callback);
//     ros::spin();
// }