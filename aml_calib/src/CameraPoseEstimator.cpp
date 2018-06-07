#include <iostream>
#include <aruco/aruco.h>
#include <aruco/cvdrawingutils.h>

#include <opencv2/core/core.hpp>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <marker_odometry/aruco_utils.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <sstream>

#include "aml_calib/MarkerPoseStamped.h"

using namespace aruco;

class CameraPoseEstimator
{
private:
  cv::Mat input_image_;
  aruco::CameraParameters cam_params_;

  bool use_rectified_images_;
  MarkerDetector marker_detector_;
  vector<Marker> detected_markers_;
  ros::Subscriber cam_info_sub_;
  bool cam_info_received_;
  image_transport::Publisher image_pub_;
  ros::Publisher marker_pose_pub_;
  ros::Publisher transform_pub_; 
  ros::Publisher cam_info_pub_;
  std::string marker_frame_;
  std::string camera_frame_;


  double marker_size_;
  int marker_id_;

  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber rgb_image_sub_;

  tf::TransformListener tf_listener_;
  tf::TransformBroadcaster tf_broadcaster_;

  sensor_msgs::CameraInfo cam_info_;

public:
  CameraPoseEstimator()
    : cam_info_received_(false),
      nh_("~"),
      it_(nh_)
  {


    nh_.param<double>("marker_size_", marker_size_, 0.05);

    nh_.param<int>("marker_id", marker_id_, 300);
    nh_.param<std::string>("camera_frame", camera_frame_, "");
    nh_.param<std::string>("marker_frame", marker_frame_, "");
    nh_.param<bool>("image_is_rectified", use_rectified_images_, true);


    cam_info_sub_ = nh_.subscribe("/rgb_camera_info", 5, &CameraPoseEstimator::rgb_cam_info_callback, this);
    rgb_image_sub_ = it_.subscribe("/rgb_rect_image", 5, &CameraPoseEstimator::rgb_image_callback, this);

    
    image_pub_ = it_.advertise("result", 0);
    transform_pub_ = nh_.advertise<geometry_msgs::TransformStamped>("transform", 100);
    marker_pose_pub_ = nh_.advertise<aml_calib::MarkerPoseStamped>("marker_pose", 100);

    cam_info_pub_ = nh_.advertise<sensor_msgs::CameraInfo>("camera_info", 100);

    


    ROS_ASSERT(camera_frame_ != "" && marker_frame_ != "");

  }

  bool getTransform(const std::string& reference_frame,
                    const std::string& child_frame,
                    tf::StampedTransform& transform)
  {
    std::string error_msg;

    if ( !tf_listener_.waitForTransform(reference_frame,
                                       child_frame,
                                       ros::Time(0),
                                       ros::Duration(0.5),
                                       ros::Duration(0.01),
                                       &error_msg)
         )
    {
      ROS_ERROR_STREAM("Unable to get pose from TF: " << error_msg);
      return false;
    }
    else
    {
      try
      {
        tf_listener_.lookupTransform( reference_frame, child_frame,
                                     ros::Time(0),                  //get latest available
                                     transform);
      }
      catch ( const tf::TransformException& e)
      {
        ROS_ERROR_STREAM("Error in lookupTransform of " << child_frame << " in " << reference_frame);
        return false;
      }

    }
    return true;
  }


  void rgb_image_callback(const sensor_msgs::ImageConstPtr& msg)
  {

    
    if(cam_info_received_)
    {
      ros::Time curr_stamp(ros::Time::now());
      cv_bridge::CvImagePtr cv_ptr;
      try
      {

        // Get image
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        input_image_ = cv_ptr->image;



        //detection results will go into "markers"
        detected_markers_.clear();

        //Ok, let's detect markers
        marker_detector_.detect(input_image_, detected_markers_, cam_params_, marker_size_, false);

        
        //for each detected marker, draw info and its boundaries in the image
        for(size_t i=0; i< detected_markers_.size(); ++i)
        {
             tf::Transform marker_pose = aruco_utils::arucoMarker2Tf(detected_markers_[i]);

             std::stringstream ss;
             ss << marker_frame_ << detected_markers_[i].id;
             tf::StampedTransform stamped_marker_pose(marker_pose, curr_stamp,
                                                  camera_frame_, ss.str().c_str());

            tf_broadcaster_.sendTransform(stamped_marker_pose);


            aml_calib::MarkerPoseStamped marker_pose_msg;

            tf::poseTFToMsg(stamped_marker_pose, marker_pose_msg.pose.pose);
            marker_pose_msg.pose.header.frame_id = camera_frame_;
            marker_pose_msg.pose.header.stamp = curr_stamp;
            marker_pose_msg.marker_id = detected_markers_[i].id;
            marker_pose_pub_.publish(marker_pose_msg);

            //draw a 3d cube in each marker if there is 3d info
            if(cam_params_.isValid() && marker_size_ != -1)
            {
              CvDrawingUtils::draw3dAxis(input_image_, detected_markers_[i], cam_params_);
              //CvDrawingUtils::draw3dCube(input_image_, detected_markers_[i], cam_params_);
            }

          


        }

        if(image_pub_.getNumSubscribers() > 0)
        {
          //show input with augmented information
          cv_bridge::CvImage out_msg;
          out_msg.header.stamp = curr_stamp;
          out_msg.encoding = sensor_msgs::image_encodings::RGB8;
          out_msg.image = input_image_;
          out_msg.header.frame_id = camera_frame_;


          sensor_msgs::CameraInfo cam_info_msg = cam_info_;
          cam_info_msg.header.stamp = curr_stamp;
          cam_info_msg.header.frame_id = camera_frame_;
          cam_info_pub_.publish(cam_info_msg);

          image_pub_.publish(out_msg.toImageMsg());
        }






      }
      catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }
    }
  }


  void rgb_cam_info_callback(const sensor_msgs::CameraInfo &msg)
  {
    cam_params_ = aruco_utils::rosCameraInfo2ArucoCamParams(msg, use_rectified_images_);

    cam_info_ = msg;

    cam_info_received_ = true;
    cam_info_sub_.shutdown();
  }
};


int main(int argc,char **argv)
{
  ros::init(argc, argv, "aml_camera_pose_estimator");

  CameraPoseEstimator node;

  ros::spin();
}
