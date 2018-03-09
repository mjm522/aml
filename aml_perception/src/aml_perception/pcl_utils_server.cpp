#include "ros/ros.h"
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
        if (cloud == 0)
        {
            res.success = false;
            res.info = "failed";
            return false;
        }


        res.out_cloud_1 = *(pcl_ros_converter_.ROSMsgFromPclCloud(*cloud));

        // res.out_cloud_1 = *out;
        res.info = req.in_string_1 + " read success";
        res.success = true;
        return true;
    }

    else if (req.function == "save_to_file")
    {
        ROS_INFO("Saving pcd file: %s", req.in_string_1.c_str());

        aml_pcloud::PointCloudPtr cloud = pcl_ros_converter_.pclCloudFromROSMsg(req.in_cloud_1);
        pcl_processor_.saveToPcdFile(req.in_string_1, cloud);

        res.success = true;
        res.info = "PCD saved to " + req.in_string_1;
        return true;
    } 


}


void initiliseServer()
{
    ros::NodeHandle nh_;
    ros::ServiceServer service_ = nh_.advertiseService("aml_pcl_service", processRequest_);

    ROS_INFO("AML PointCloud Utility Server Running...\nAvailable Services:\n   <request.function>  <args> \t\t||\t <responses> \n\n1. \"read_pcd_file\"  \"[file name]\"\t||\t [sensor_msgs/PointCloud2 out_cloud_1] \t[string info] \t[bool success]\n2. \"save_to_file\" [sensor_msgs/PointCloud2 in_cloud_1]  \"[file name]\"\t|| \t[string info] \t[bool success]");

    ros::spin();
};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "pcl_utility_server");

    initiliseServer();


    // PCLUtilityServer server;

    return 0;
};