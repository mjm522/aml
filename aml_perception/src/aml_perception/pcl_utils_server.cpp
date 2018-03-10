#include "ros/ros.h"
#include <string>
#include <sstream>
#include "pcl_processing.h"
#include "aml_services/PCLUtility.h"


aml_pcloud::PclRosConversions pcl_ros_converter_;
aml_pcloud::PCLProcessor pcl_processor_;

// ----- starts the ros service
void initiliseServer();

// ----- processes the service request and calls the required method from pcl_processing
bool processRequest_(aml_services::PCLUtility::Request  &req,
         aml_services::PCLUtility::Response &res);


bool processRequest_(aml_services::PCLUtility::Request  &req,
         aml_services::PCLUtility::Response &res)
{
    if (req.function == "read_pcd_file")
    {
        ROS_INFO("Reading pcd file: %s", req.in_string_1.c_str());


        aml_pcloud::PointCloudPtr cloud = pcl_processor_.getCloudFromPcdFile(req.in_string_1);
        if (cloud == NULL)
        {
            res.success = false;
            res.info = "failed";
            ROS_ERROR("Failed to read from PCD file");
            return false;
        }


        res.out_cloud_1 = pcl_ros_converter_.ROSMsgFromPclCloud(*cloud);

        // res.out_cloud_1 = *out;
        res.info = req.in_string_1 + " read success";
        res.success = true;
        return true;
    }

    else if (req.function == "save_to_file")
    {
        if (req.in_string_1.empty())
        {
            res.success = false;
            res.info = "failed";
            ROS_ERROR("Destination path not specified in in_string_1");
            return false;
        }
        ROS_INFO("Saving pcd file: %s", req.in_string_1.c_str());

        aml_pcloud::PointCloudPtr cloud = pcl_ros_converter_.pclCloudFromROSMsg(req.in_cloud_1);
        pcl_processor_.saveToPcdFile(req.in_string_1, cloud);

        res.success = true;
        res.info = "PCD saved to " + req.in_string_1;
        return true;
    }

    else if (req.function == "downsample_cloud")
    {
        
        aml_pcloud::PointCloudPtr cloud_in = pcl_ros_converter_.pclCloudFromROSMsg(req.in_cloud_1);

        aml_pcloud::PointCloudPtr cloud_out = pcl_processor_.downsampleCloud(cloud_in, req.in_float_array_1);

        std::ostringstream info ;
        info  << "("<< req.in_float_array_1[0] << ", " << req.in_float_array_1[1] << ", " << req.in_float_array_1[2] << ")";
        std::string info_str = info.str();
        ROS_INFO("Downsampling cloud with leaf sizes: %s", info_str.c_str());
        // std::cout << info_str << std::endl;

        res.out_cloud_1 = pcl_ros_converter_.ROSMsgFromPclCloud(*cloud_out);

        std::ostringstream stm ;
        stm << "downsampled with leafsize " << "("<< req.in_float_array_1[0] << ", " << req.in_float_array_1[1] << ", " << req.in_float_array_1[2] << ")";
        res.info = stm.str();
        res.success = true;
        return true;
    }

}


void initiliseServer()
{
    ros::NodeHandle nh_;
    ros::ServiceServer service_ = nh_.advertiseService("aml_pcl_service", processRequest_);

    ROS_INFO("AML PointCloud Utility Server Running...\nAvailable Services:\n   <request.function>  <args> \t\t||\t <responses> \n\n1. \"read_pcd_file\"  \"[file name]\"\t||\t [sensor_msgs/PointCloud out_cloud_1] [string info] [bool success]\n2. \"save_to_file\" [sensor_msgs/PointCloud in_cloud_1]  \"[file name]\"\t||\t [string info] [bool success]\n3. \"downsample_cloud\" [sensor_msgs/PointCloud in_cloud_1] {float_array in_float_array_1 (leafsize)}\t||\t [sensor_msgs/PointCloud out_cloud_1] [string info] [bool success]");

    ros::spin();
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "pcl_utility_server");

    initiliseServer();


    // PCLUtilityServer server;

    return 0;
}