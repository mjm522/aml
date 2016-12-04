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

using namespace aruco;

class MarkerOdometry
{
private:
  cv::Mat inImage;
  aruco::CameraParameters right_hand_camParam, openni_rgb_camParam;

  bool useRectifiedImages;
  MarkerDetector right_hand_mDetector, openni_rgb_mDetector;
  vector<Marker> markers;
  ros::Subscriber right_hand_cam_info_sub, openni_rgb_cam_info_sub;
  bool right_hand_cam_info_received, openni_rgb_cam_info_received, calibrated, computedMarkerToBase;
  image_transport::Publisher image_pub;
  image_transport::Publisher debug_pub;
  ros::Publisher pose_pub;
  ros::Publisher transform_pub; 
  ros::Publisher position_pub;
  std::string marker_frame;
  std::string camera_frame;
  std::string reference_frame;
  tf::Transform transformOpenniToBase, transformLeftHandMarkerToBase, transformBoxToBase;

  double left_hand_marker_size, box_marker_size;
  int left_hand_marker_id, box_marker_id;

  ros::NodeHandle nh;
  image_transport::ImageTransport it;
  image_transport::Subscriber right_hand_image_sub, openni_rgb_image_sub;

  tf::TransformListener _tfListener;
  tf::TransformBroadcaster br;

public:
  MarkerOdometry()
    : right_hand_cam_info_received(false), openni_rgb_cam_info_received(false), calibrated(false),
      nh("~"), computedMarkerToBase(false),
      it(nh)
  {
  
    right_hand_cam_info_sub = nh.subscribe("/right_hand_camera_info", 1, &MarkerOdometry::right_hand_cam_info_callback, this);
    right_hand_image_sub = it.subscribe("/right_hand_image", 1, &MarkerOdometry::right_hand_image_callback, this);
    
    openni_rgb_cam_info_sub = nh.subscribe("/openni_rgb_camera_info", 1, &MarkerOdometry::openni_rgb_cam_info_callback, this);
    openni_rgb_image_sub = it.subscribe("/openni_rgb_rect_image", 1, &MarkerOdometry::openni_rgb_image_callback, this);
    
    image_pub = it.advertise("result", 0);
    debug_pub = it.advertise("debug", 0);
    pose_pub = nh.advertise<geometry_msgs::PoseStamped>("pose", 100);
    transform_pub = nh.advertise<geometry_msgs::TransformStamped>("transform", 100);
    position_pub = nh.advertise<geometry_msgs::Vector3Stamped>("position", 100);

    nh.param<double>("left_hand_marker_size", left_hand_marker_size, 0.05);
    nh.param<double>("box_marker_size", box_marker_size, 0.05);
    nh.param<int>("left_hand_marker_id", left_hand_marker_id, 300);
    nh.param<int>("box_marker_id", box_marker_id, 300);
    nh.param<std::string>("reference_frame", reference_frame, "");
    nh.param<std::string>("camera_frame", camera_frame, "");
    nh.param<std::string>("marker_frame", marker_frame, "");
    nh.param<bool>("image_is_rectified", useRectifiedImages, true);


    ROS_ASSERT(camera_frame != "" && marker_frame != "");

    if ( reference_frame.empty() )
      reference_frame = camera_frame;

    ROS_INFO("Aruco node started with marker size of %f m and marker id to track: %d",
             left_hand_marker_size, left_hand_marker_id);
    ROS_INFO("Aruco node started with marker size of %f m and marker id to track: %d",
             box_marker_size, box_marker_id);
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


  void right_hand_image_callback(const sensor_msgs::ImageConstPtr& msg)
  {
    if (computedMarkerToBase)
    {
      //this makes sure that right_hand_image_callback is only called till 
      // the left hand marker to base is calculated.
      right_hand_image_sub.shutdown();
      return;
    }
    
    if(right_hand_cam_info_received)
    {
      
      ros::Time curr_stamp(ros::Time::now());
      cv_bridge::CvImagePtr cv_ptr;
      try
      {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        inImage = cv_ptr->image;

        //detection results will go into "markers"
        markers.clear();
        //Ok, let's detect
        right_hand_mDetector.detect(inImage, markers, right_hand_camParam, left_hand_marker_size, false);
        //for each marker, draw info and its boundaries in the image
        for(size_t i=0; i<markers.size(); ++i)
        {
          // only publishing the selected marker
          if(markers[i].id == left_hand_marker_id)
          {

            tf::Transform transformRightHandCamToMarker = aruco_utils::arucoMarker2Tf(markers[i]);
            tf::StampedTransform rightHandCameraToBase;

            getTransform("base",
                         "right_hand_camera",
                           rightHandCameraToBase);

            transformLeftHandMarkerToBase = (static_cast<tf::Transform>(rightHandCameraToBase)*transformRightHandCamToMarker);
 
            //this would be only transmitted once.
            tf::StampedTransform stampedTransform(transformLeftHandMarkerToBase, curr_stamp,
                                                  "base", "leftHandMarker");
            br.sendTransform(stampedTransform);

            geometry_msgs::PoseStamped poseMsg;
            tf::poseTFToMsg(transformRightHandCamToMarker, poseMsg.pose);
            poseMsg.header.frame_id = reference_frame;
            poseMsg.header.stamp = curr_stamp;
            pose_pub.publish(poseMsg);

            geometry_msgs::TransformStamped transformMsg;
            tf::transformStampedTFToMsg(stampedTransform, transformMsg);
            transform_pub.publish(transformMsg);

            geometry_msgs::Vector3Stamped positionMsg;
            positionMsg.header = transformMsg.header;
            positionMsg.vector = transformMsg.transform.translation;
            position_pub.publish(positionMsg);
           
            computedMarkerToBase = true;
          }

          // but drawing all the detected markers
          markers[i].draw(inImage,cv::Scalar(0,0,255),2);
        }

        //draw a 3d cube in each marker if there is 3d info
        if(right_hand_camParam.isValid() && left_hand_marker_size!=-1)
        {
          CvDrawingUtils::draw3dAxis(inImage, markers[0], right_hand_camParam);
        }

        if(image_pub.getNumSubscribers() > 0)
        {
          //show input with augmented information
          cv_bridge::CvImage out_msg;
          out_msg.header.stamp = curr_stamp;
          out_msg.encoding = sensor_msgs::image_encodings::RGB8;
          out_msg.image = inImage;
          image_pub.publish(out_msg.toImageMsg());
        }

        if(debug_pub.getNumSubscribers() > 0)
        {
          //show also the internal image resulting from the threshold operation
          cv_bridge::CvImage debug_msg;
          debug_msg.header.stamp = curr_stamp;
          debug_msg.encoding = sensor_msgs::image_encodings::MONO8;
          debug_msg.image = right_hand_mDetector.getThresholdedImage();
          debug_pub.publish(debug_msg.toImageMsg());
        }

      }
      catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }
    }
  }

  void openni_rgb_image_callback(const sensor_msgs::ImageConstPtr& msg)
  {
    // wait until object pose is computed with respect to right hand
    if(!computedMarkerToBase)
    {
      //come after the right hand data is available and 
      // has been successfully computed
      return;
    }
    
    if(openni_rgb_cam_info_received)
    {
      ros::Time curr_stamp(ros::Time::now());
      cv_bridge::CvImagePtr cv_ptr;
      try
      {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        inImage = cv_ptr->image;

        //detection results will go into "markers"
        markers.clear();
        //Ok, let's detect
        if (calibrated)
        {
          openni_rgb_mDetector.detect(inImage, markers, openni_rgb_camParam, box_marker_size, false);
        }
        else
        {
          openni_rgb_mDetector.detect(inImage, markers, openni_rgb_camParam, left_hand_marker_size, false);
        }
        
        //for each marker, draw info and its boundaries in the image
        for(size_t i=0; i<markers.size(); ++i)
        {
          //for left hand marker, called till calibrated
          if((markers[i].id == left_hand_marker_id) && (!calibrated)) //
          {
           
            tf::Transform transformOpenniToMarker = aruco_utils::arucoMarker2Tf(markers[i]);
            tf::StampedTransform leftHandMarkerToBase;

            getTransform("base",
                         "leftHandMarker",
                         leftHandMarkerToBase);  

            transformOpenniToBase = (static_cast<tf::Transform>(leftHandMarkerToBase)*transformOpenniToMarker.inverse());

            calibrated = true;

          }
          
          //for box marker
          if((markers[i].id == box_marker_id) && (calibrated)) 
          {

            //keep in mind that this is called only if the marker on the box
            // is visible
            //only perform the following operations if we know the pose of openni camera w.r.t base
            tf::Transform transformBoxToOpenni = aruco_utils::arucoMarker2Tf(markers[i]);

            transformBoxToBase = (static_cast<tf::Transform>(transformOpenniToBase)*transformBoxToOpenni);

            tf::StampedTransform stampedTransformBoxToBase(transformBoxToBase, curr_stamp,
                                                  "base", "box");
            br.sendTransform(stampedTransformBoxToBase);

            //draw a 3d cube in each marker if there is 3d info
            if(openni_rgb_camParam.isValid() && box_marker_size != -1)
            {
              CvDrawingUtils::draw3dAxis(inImage, markers[i], openni_rgb_camParam);
              //CvDrawingUtils::draw3dCube(inImage, markers[1], openni_rgb_camParam);
            }

          }

        }

       if(calibrated)
        {
          // in case the box is not visible,
          // but still if the arm is calibrated
          // send openni camera location
          tf::StampedTransform stampedTransformOpenniToBase(transformOpenniToBase, curr_stamp,
                                                "base", "openni_rgb_camera");
          br.sendTransform(stampedTransformOpenniToBase);
        }


        if(image_pub.getNumSubscribers() > 0)
        {
          //show input with augmented information
          cv_bridge::CvImage out_msg;
          out_msg.header.stamp = curr_stamp;
          out_msg.encoding = sensor_msgs::image_encodings::RGB8;
          out_msg.image = inImage;
          image_pub.publish(out_msg.toImageMsg());
        }

      }
      catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }
    }
  }

  // wait for one camerainfo, then shut down that subscriber
  void right_hand_cam_info_callback(const sensor_msgs::CameraInfo &msg)
  {
    right_hand_camParam = aruco_utils::rosCameraInfo2ArucoCamParams(msg, useRectifiedImages);

    right_hand_cam_info_received = true;
    right_hand_cam_info_sub.shutdown();
  }

  void openni_rgb_cam_info_callback(const sensor_msgs::CameraInfo &msg)
  {
    openni_rgb_camParam = aruco_utils::rosCameraInfo2ArucoCamParams(msg, useRectifiedImages);

    openni_rgb_cam_info_received = true;
    openni_rgb_cam_info_sub.shutdown();
  }
};


int main(int argc,char **argv)
{
  ros::init(argc, argv, "aml_marker_odometry");

  MarkerOdometry node;

  ros::spin();
}
