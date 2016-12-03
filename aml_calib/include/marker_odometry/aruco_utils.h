#ifndef _ARUCO_ILS_H_
#define _ARUCO_ILS_H_

#include <aruco/aruco.h>
#include <sensor_msgs/CameraInfo.h>
#include <tf/transform_datatypes.h>

namespace aruco_utils
{

  /**
     * @brief rosCameraInfo2ArucoCamParams gets the camera intrinsics from a CameraInfo message and copies them
     *                                     to aruco_ros own data structure
     * @param cam_info
     * @param useRectifiedParameters if true, the intrinsics are taken from cam_info.P and the distortion parameters
     *                               are set to 0. Otherwise, cam_info.K and cam_info.D are taken.
     * @return
     */
  aruco::CameraParameters rosCameraInfo2ArucoCamParams(const sensor_msgs::CameraInfo& cam_info,
                                                       bool useRectifiedParameters);

  tf::Transform arucoMarker2Tf(const aruco::Marker& marker);

}
#endif // _ARUCO_ILS_H_
