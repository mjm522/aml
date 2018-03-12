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
        ROS_INFO("Reading pcd file: %s", req.msg_in.string_1.c_str());


        aml_pcloud::PointCloudPtr cloud = pcl_processor_.getCloudFromPcdFile(req.msg_in.string_1);
        if (cloud == NULL)
        {
            res.success = false;
            res.info = "failed";
            ROS_ERROR("Failed to read from PCD file");
            return false;
        }


        res.msg_out.cloud_1 = pcl_ros_converter_.ROSMsgFromPclCloud(*cloud);

        // res.out_cloud_1 = *out;
        res.info = req.msg_in.string_1 + " read success";
        res.success = true;
        return true;
    }

    else if (req.function == "save_to_file")
    {
        if (req.msg_in.string_1.empty())
        {
            res.success = false;
            res.info = "Failed: Destination path not specified in in_string_1.";
            ROS_ERROR("Destination path not specified!");
            return false;
        }
        ROS_INFO("Saving pcd file: %s", req.msg_in.string_1.c_str());

        aml_pcloud::PointCloudPtr cloud = pcl_ros_converter_.pclCloudFromROSMsg(req.msg_in.cloud_1);
        pcl_processor_.saveToPcdFile(req.msg_in.string_1, cloud);

        res.success = true;
        res.info = "PCD saved to " + req.msg_in.string_1;
        return true;
    }

    else if (req.function == "downsample_cloud")
    {
        
        aml_pcloud::PointCloudPtr cloud_in = pcl_ros_converter_.pclCloudFromROSMsg(req.msg_in.cloud_1);

        aml_pcloud::PointCloudPtr cloud_out = pcl_processor_.downsampleCloud(cloud_in, req.msg_in.float_array_1);

        std::ostringstream info ;
        info  << "("<< req.msg_in.float_array_1[0] << ", " << req.msg_in.float_array_1[1] << ", " << req.msg_in.float_array_1[2] << ")";
        std::string info_str = info.str();
        ROS_INFO("Downsampling cloud with leaf sizes: %s", info_str.c_str());
        // std::cout << info_str << std::endl;

        res.msg_out.cloud_1 = pcl_ros_converter_.ROSMsgFromPclCloud(*cloud_out);

        std::ostringstream stm;
        stm << "downsampled with leafsize " << "("<< req.msg_in.float_array_1[0] << ", " << req.msg_in.float_array_1[1] << ", " << req.msg_in.float_array_1[2] << ")";
        res.info = stm.str();
        res.success = true;
        return true;
    }

    else if (req.function == "get_curvature" or req.function == "fit_plane" or req.function == "compute_point_normal")
    {
        aml_pcloud::PointCloudPtr cloud_in = pcl_ros_converter_.pclCloudFromROSMsg(req.msg_in.cloud_1);

        if (req.msg_in.int_array_1.size() > 0) ROS_INFO("Computing plane parameters and normals using the provided point indices");
        else ROS_INFO("Computing plane parameters and normals using full cloud");

        pcl_processor_.fitPlaneAndGetCurvature(cloud_in, req.msg_in.int_array_1, res.msg_out.float_array_1, res.msg_out.float_1);

        res.info = "Plane parameters in msg_out.float_array_1; curvature in msg_out.float_1";
        res.success = true;
        return true;
    }

    else if (req.function == "compute_all_normals")
    {
        aml_pcloud::PointCloudPtr cloud_in = pcl_ros_converter_.pclCloudFromROSMsg(req.msg_in.cloud_1);
        ROS_INFO("Computing normals for all points in plane");


        aml_pcloud::PointCloudPtr cloud_out = pcl_processor_.computeNormalForAllPoints(cloud_in);

        res.msg_out.cloud_1 = pcl_ros_converter_.ROSMsgFromPclCloud(*cloud_out);

        res.info = "Normals stored in the format [normal_x, normal_y, normal_z] in msg_out.cloud_1";
        res.success = true;
        return true;

    }

    else if (req.function == "apply_transformation")
    {
        aml_pcloud::PointCloudPtr cloud_in = pcl_ros_converter_.pclCloudFromROSMsg(req.msg_in.cloud_1);
        ROS_INFO("Transforming point cloud");

        aml_pcloud::PointCloudPtr cloud_out = pcl_processor_.transformPointCloud(cloud_in, req.msg_in.float_array_1);

        res.msg_out.cloud_1 = pcl_ros_converter_.ROSMsgFromPclCloud(*cloud_out);

        res.info = "Transformed cloud in msg_out.cloud_1";
        res.success = true;
        return true;

    }

    else if (req.function == "add_clouds")
    {
        aml_pcloud::PointCloudPtr cloud_in_1 = pcl_ros_converter_.pclCloudFromROSMsg(req.msg_in.cloud_1);
        aml_pcloud::PointCloudPtr cloud_in_2 = pcl_ros_converter_.pclCloudFromROSMsg(req.msg_in.cloud_2);
        ROS_INFO("Adding two clouds");

        aml_pcloud::PointCloudPtr cloud_out = pcl_processor_.addPointClouds(cloud_in_1, cloud_in_2);

        res.msg_out.cloud_1 = pcl_ros_converter_.ROSMsgFromPclCloud(*cloud_out);

        res.info = "Concatenated cloud in msg_out.cloud_1";
        res.success = true;
        return true;

    }

    else if (req.function == "get_points_not_in_plane")
    {
        aml_pcloud::PointCloudPtr cloud_in = pcl_ros_converter_.pclCloudFromROSMsg(req.msg_in.cloud_1);
        ROS_INFO("Fitting points to plane and computing outliers");

        aml_pcloud::PointCloudPtr cloud_out = pcl_processor_.getPointsNotInPlane(cloud_in);

        res.msg_out.cloud_1 = pcl_ros_converter_.ROSMsgFromPclCloud(*cloud_out);

        res.info = "Outliers-cloud in msg_out.cloud_1";
        res.success = true;
        return true;

    }

    else if (req.function == "compute_centroid")
    {
        aml_pcloud::PointCloudPtr cloud_in = pcl_ros_converter_.pclCloudFromROSMsg(req.msg_in.cloud_1);
        ROS_INFO("Computing centroid of cloud");

        res.msg_out.float_array_1 = pcl_processor_.computeCentroid(cloud_in);

        res.info = "Centroid in msg_out.float_array_1";
        res.success = true;
        return true;

    }


}


void initiliseServer()
{
    ros::NodeHandle nh_;
    ros::ServiceServer service_ = nh_.advertiseService("aml_pcl_service", processRequest_);

    ROS_INFO("AML PointCloud Utility Server Running...\nAll responses have 'info (string)' and 'success (bool)' parameters...\nAvailable Services:\n   [request.function]  [required args] {optional args} \t\t||\t <responses> \n\n1. \"read_pcd_file\"  [msg_in.string_1 (file name)]\t||\t [sensor_msgs/PointCloud msg_out.cloud_1]\n2. \"save_to_file\" [sensor_msgs/PointCloud msg_in.cloud_1]  [msg_in.string_1 (file name)]\t||\t\n3. \"downsample_cloud\" [sensor_msgs/PointCloud msg_in.cloud_1] {float_array msg_in.float_array_1 (leafsize)}\t||\t [sensor_msgs/PointCloud msg_out.cloud_1]\n4. \"compute_point_normal/get_curvature/fit_plane\" [sensor_msgs/PointCloud msg_in.cloud_1] {int_array msg_in.int_array_1 (indices)}\t||\t [float_array msg_out.float_array_1 (plane parameters)] [float curvature]\n5. \"compute_all_normals\" [sensor_msgs/PointCloud msg_in.cloud_1]\t||\t [sensor_msgs/PointCloud msg_out.cloud_1]\n6. \"apply_transformation\" [sensor_msgs/PointCloud msg_in.cloud_1] [float_array msg_in.float_array_1 (flattened trans matrix)]\t||\t [sensor_msgs/PointCloud msg_out.cloud_1]\n7. \"add_cloud\" [sensor_msgs/PointCloud msg_in.cloud_1] [sensor_msgs/PointCloud msg_in.cloud_2]\t||\t [sensor_msgs/PointCloud msg_out.cloud_1]\n8. \"get_points_not_in_plane\" [sensor_msgs/PointCloud msg_in.cloud_1]\t||\t [sensor_msgs/PointCloud msg_out.cloud_1]\n9. \"compute_centroid\" [sensor_msgs/PointCloud msg_in.cloud_1]\t||\t [float_array msg_out.float_array_1 (x,y,z)]");

    ros::spin();
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "pcl_utility_server");

    initiliseServer();


    // PCLUtilityServer server;

    return 0;
}