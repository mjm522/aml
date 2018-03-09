#include "ros/ros.h"
#include "pcl_processing.h"
#include "aml_services/PCLCustomMsg.h"


class PCLUtilityServer
{

private:
    aml_pcloud::PclRosConversions::ConversionPtr pcl_ros_converter_;
    ros::NodeHandle nh_;
    ros::ServiceServer service_;

    void initiliseServer_();

    static bool processRequest(aml_services::PCLCustomMsg::Request  &req,
         aml_services::PCLCustomMsg::Response &res)
{
  return true;
}


public:
    PCLUtilityServer() : pcl_ros_converter_(new aml_pcloud::PclRosConversions) 
    {
        this->initiliseServer_();
    };


};

void PCLUtilityServer::initiliseServer_()
{
    service_ = nh_.advertiseService("aml_pcl_service", this->processRequest);
    // ros::ServiceServer service2 = n.advertiseService("sub_two_ints", add);
    ROS_INFO("AML PointCloud Utility Server Running...");
    ros::spin();
};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "pcl_utility_server");
    PCLUtilityServer server;

    return 0;
};