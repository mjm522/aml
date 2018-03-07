#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_2D.h>
#include <pcl/sample_consensus/sac_model_registration.h>
#include <pcl/sample_consensus/ransac.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/random_sample.h>

typedef pcl::PointXYZ CloudPoint;