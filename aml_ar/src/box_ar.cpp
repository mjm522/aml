#include <iostream>
#include <aruco/aruco.h>
#include <aruco/cvdrawingutils.h>

#include <opencv2/core/core.hpp>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
// #include <marker_odometry/aruco_utils.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

using namespace aruco;


void tf2Mat(tf::Transform& transform, cv::Mat& rmat){
    tf::Matrix3x3 rot(transform.getRotation());
    rmat = cv::Mat(3,3,CV_32FC1);


    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            rmat.at<float>(i,j) = rot[i][j];
        }
    }



}

void tf2Mat(tf::Transform& transform, cv::Mat& tvec, cv::Mat& rvec){

  tvec = cv::Mat(1,3,CV_32FC1);
  rvec = cv::Mat(1,3,CV_32FC1);
  cv::Mat rmat = cv::Mat(3,3,CV_32FC1);



  tf::Matrix3x3 rot(transform.getRotation());
  tf::Vector3 orig = transform.getOrigin();

  cv::Mat rotate_to_ros(3, 3, CV_32FC1);
    // -1 0 0
    // 0 0 1
    // 0 1 0
  rotate_to_ros.at<float>(0,0) = -1.0;
  rotate_to_ros.at<float>(0,1) = 0.0;
  rotate_to_ros.at<float>(0,2) = 0.0;
  rotate_to_ros.at<float>(1,0) = 0.0;
  rotate_to_ros.at<float>(1,1) = 0.0;
  rotate_to_ros.at<float>(1,2) = 1.0;
  rotate_to_ros.at<float>(2,0) = 0.0;
  rotate_to_ros.at<float>(2,1) = 1.0;
  rotate_to_ros.at<float>(2,2) = 0.0;




  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 3; j++){
      rmat.at<float>(i,j) = rot[i][j];
    }
  }

  // rmat = rmat*rotate_to_ros.t();

  // for(int i = 0; i < 3; i++)
  //   tvec.at<float>(i) = orig[i];


  tvec.at<float>(0) = orig[0];
  tvec.at<float>(1) = orig[1];
  tvec.at<float>(2) = orig[2];

  
  cv::Rodrigues(rmat, rvec);


}


aruco::CameraParameters rosCameraInfo2ArucoCamParams(const sensor_msgs::CameraInfo& cam_info,
                                                                bool useRectifiedParameters)
{
    cv::Mat cameraMatrix(3, 3, CV_32FC1);
    cv::Mat distorsionCoeff(4, 1, CV_32FC1);
    cv::Size size(cam_info.height, cam_info.width);

    if ( useRectifiedParameters )
    {
      cameraMatrix.setTo(0);
      cameraMatrix.at<float>(0,0) = cam_info.P[0];   cameraMatrix.at<float>(0,1) = cam_info.P[1];   cameraMatrix.at<float>(0,2) = cam_info.P[2];
      cameraMatrix.at<float>(1,0) = cam_info.P[4];   cameraMatrix.at<float>(1,1) = cam_info.P[5];   cameraMatrix.at<float>(1,2) = cam_info.P[6];
      cameraMatrix.at<float>(2,0) = cam_info.P[8];   cameraMatrix.at<float>(2,1) = cam_info.P[9];   cameraMatrix.at<float>(2,2) = cam_info.P[10];

      for(int i=0; i<4; ++i)
        distorsionCoeff.at<float>(i, 0) = 0;
    }
    else
    {
      for(int i=0; i<9; ++i)
        cameraMatrix.at<float>(i%3, i-(i%3)*3) = cam_info.K[i];

      if(cam_info.D.size() == 4)
      {
        for(int i=0; i<4; ++i)
          distorsionCoeff.at<float>(i, 0) = cam_info.D[i];
      }
      else
      {
        ROS_WARN("length of camera_info D vector is not 4, assuming zero distortion...");
        for(int i=0; i<4; ++i)
          distorsionCoeff.at<float>(i, 0) = 0;
      }
    }

    return aruco::CameraParameters(cameraMatrix, distorsionCoeff, size);
}

class BoxAR
{
private:
  cv::Mat inImage;
  aruco::CameraParameters openni_rgb_camParam;

  bool useRectifiedImages;

  ros::Subscriber hand_cam_info_sub, openni_rgb_cam_info_sub;
  bool hand_cam_info_received, openni_rgb_cam_info_received, received_cam_pose, computedMarkerToBase, received_box_goal;
  image_transport::Publisher image_pub;
  image_transport::Publisher debug_pub;
  ros::Publisher pose_pub;
  ros::Publisher transform_pub; 
  ros::Publisher position_pub;
  std::string marker_frame;
  std::string camera_frame;
  std::string reference_frame;
  tf::Transform transformOpenniToBase, transformBoxToBase;

  double box_marker_size;

  ros::NodeHandle nh;
  image_transport::ImageTransport it;
  image_transport::Subscriber openni_rgb_image_sub;

  tf::TransformListener _tfListener;
  tf::TransformBroadcaster br;

  tf::StampedTransform baseToOpenni;
  tf::StampedTransform boxGoalTransform;


public:
  BoxAR()
    : openni_rgb_cam_info_received(false), received_cam_pose(false),
      nh("~"), computedMarkerToBase(false), received_box_goal(false),
      it(nh)
  {

    nh.param<double>("box_marker_size", box_marker_size, 0.05);
    nh.param<std::string>("reference_frame", reference_frame, "");
    nh.param<std::string>("camera_frame", camera_frame, "");
    nh.param<std::string>("marker_frame", marker_frame, "");
    nh.param<bool>("image_is_rectified", useRectifiedImages, true);

    openni_rgb_cam_info_sub = nh.subscribe("/openni_rgb_camera_info", 5, &BoxAR::openni_rgb_cam_info_callback, this);
    openni_rgb_image_sub = it.subscribe("/openni_rgb_rect_image", 5, &BoxAR::openni_rgb_image_callback, this);
    
    image_pub = it.advertise("result_ar", 0);



    ROS_ASSERT(camera_frame != "" && marker_frame != "");

    if ( reference_frame.empty() )
      reference_frame = camera_frame;

   

    ROS_INFO("Aruco node will publish pose to TF with %s as parent and %s as child.",
             reference_frame.c_str(), marker_frame.c_str());
  }

  bool getTransform(const std::string& refFrame,
                    const std::string& childFrame,
                    tf::StampedTransform& transform)
  {
    std::string errMsg;

    if ( !_tfListener.waitForTransform(refFrame,
                                       childFrame,
                                       ros::Time(0),
                                       ros::Duration(0.5),
                                       ros::Duration(0.01),
                                       &errMsg)
         )
    {
      ROS_ERROR_STREAM("Unable to get pose from TF: " << errMsg);
      return false;
    }
    else
    {
      try
      {
        _tfListener.lookupTransform( refFrame, childFrame,
                                     ros::Time(0),                  //get latest available
                                     transform);
      }
      catch ( const tf::TransformException& e)
      {
        ROS_ERROR_STREAM("Error in lookupTransform of " << childFrame << " in " << refFrame);
        return false;
      }

    }
    return true;
  }



  void openni_rgb_image_callback(const sensor_msgs::ImageConstPtr& msg)
  {
    // wait until object pose is computed with respect to right hand
    //if(!received_cam_pose){
      received_cam_pose = getTransform("openni_rgb_camera",
                                       "base",
                                        baseToOpenni); 

      ROS_INFO("RECEIVED CAMERA POSE");

    //}

    if(!received_box_goal){

      received_box_goal = getTransform("base" ,
                           "box_goal",
                           boxGoalTransform); 

      ROS_INFO("RECEIVED BOX GOAL");

    }
    
    if(openni_rgb_cam_info_received && received_cam_pose && received_box_goal)
    {

      ros::Time curr_stamp(ros::Time::now());
      cv_bridge::CvImagePtr cv_ptr;
      try
      {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        inImage = cv_ptr->image;

        box_overlay(inImage);


      }
      catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }
    }
  }

  void openni_rgb_cam_info_callback(const sensor_msgs::CameraInfo &msg)
  {
    openni_rgb_camParam = rosCameraInfo2ArucoCamParams(msg, useRectifiedImages);

    openni_rgb_cam_info_received = true;
    openni_rgb_cam_info_sub.shutdown();
  }


  void box_overlay(cv::Mat& image){


     //draw a 3d cube in each marker if there is 3d info
    if(openni_rgb_camParam.isValid())
    {

      cv::Mat tvec, rvec;
      cv::Mat box_pos(1,3,CV_32FC1);
      box_pos.at<float>(0) = boxGoalTransform.getOrigin()[0];
      box_pos.at<float>(1) = boxGoalTransform.getOrigin()[1];
      box_pos.at<float>(2) = boxGoalTransform.getOrigin()[2]; // Position of the box in the base frame of the robot


      cv::Mat box_ori = cv::Mat::eye(3,3, CV_32FC1);
      tf2Mat(boxGoalTransform,box_ori);
      
      tf2Mat(baseToOpenni, tvec, rvec);
      CvDrawingUtils::draw3dCube(image, box_pos, box_ori, tvec, rvec, box_marker_size, openni_rgb_camParam);
      ROS_INFO("GOOD STUFF 2");

               

                

     }


     if(image_pub.getNumSubscribers() > 0)
     {
        ros::Time curr_stamp(ros::Time::now());
        //show input with augmented information
        cv_bridge::CvImage out_msg;
        out_msg.header.stamp = curr_stamp;
        out_msg.encoding = sensor_msgs::image_encodings::RGB8;
        out_msg.image = image;
        image_pub.publish(out_msg.toImageMsg());
        ROS_INFO("GOOD STUFF 3");
      }

    

  }



};


int main(int argc,char **argv)
{
  ros::init(argc, argv, "aml_box_ar");

  BoxAR node;

  ros::spin();
}
