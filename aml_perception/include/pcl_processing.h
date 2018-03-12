
// All input, output types for point clouds are sensor_msg::PointCloud or pcl::PointCloud


#include <pcl/point_cloud.h>  
#include <pcl/point_types.h>  


#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/random_sample.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>

#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud_conversion.h>


namespace aml_pcloud
{
    typedef pcl::PointXYZ CloudPoint; // A point structure denoting Euclidean xyz coordinates
    typedef pcl::PointCloud<CloudPoint> PointCloud;
    typedef pcl::PointCloud<CloudPoint>::Ptr PointCloudPtr;

    typedef pcl::PointXYZRGB PointT; // A point structure denoting xyz and rgb
    typedef pcl::PointCloud<PointT> PointCloudT;
    typedef pcl::PointCloud<PointT>::Ptr PointCloudTPtr;

    typedef pcl::PCLPointCloud2 PointCloud2;


    class {
    public:
        template<typename T>
        operator boost::shared_ptr<T>() { return boost::shared_ptr<T>(); }
    } nullptr;

    class PclRosConversions
    {
    public:



        PclRosConversions() {}

    /**
     * Helper function to convert from sensor_msgs/PointCloud2 to pcl::PointCloud<pcl::PointXYZRGB>
     */
        PointCloudPtr pclCloudFromROSMsg(const sensor_msgs::PointCloud msg);

    /**
     *----- Helper function to convert from pcl::PointCloud<pcl::PointXYZ> to sensor_msgs/PointCloud2
     */
        sensor_msgs::PointCloud ROSMsgFromPclCloud(PointCloud& cloud);
  

    };

    class PCLProcessor
    {

    public:

        PCLProcessor() {}

        // ----- service request.function =  "read_pcd_file"
        PointCloudPtr getCloudFromPcdFile(std::string& input_file);

        // ----- service request.function =  "save_to_file"
        void saveToPcdFile(std::string filename, const PointCloudPtr cloud);


        PointCloudPtr downsampleCloud(const PointCloudPtr cloud, std::vector<float> &leaf_sizes); // === CAUSES SEGFAULT !!

        PointCloudPtr getPointsNotInPlane(PointCloudPtr input_cloud); // === CAUSES SEGFAULT !!

        /**
         *  computeNormalForAllPoints function
         *
         *  Computes the normals for all points in a given point cloud. 
         *
         *  
         *  @param  cloud
         *  @return cloud of normals
         */
        PointCloudPtr computeNormalForAllPoints(PointCloudPtr cloud);

        /**
         * fitPlaneAndGetCurvature function
         * Compute the Least-Squares plane fit for a given set of points, using their indices, and return the estimated plane parameters together with the surface curvature.
         * @param cloud
         * @param indices <optional>    the indices of the cloud points to be used for computation
         * 
         * @param plane parameters      the plane parameters as: a, b, c, d (ax + by + cz + d = 0)
         * @param curvature             the estimated surface curvature as a measure of lambda_0/(lambda_0 + lambda_1 + lambda_2)
         */
        void fitPlaneAndGetCurvature(const PointCloudPtr cloud, std::vector< int > indices, std::vector< float > &plane_parameters, float &curvature);
        
        /**
         *  transformPointCloud function
         *
         *  This function takes a point cloud and transforms it according to the transformation defined in trans_mat
         *  @param  input_cloud       the cloud
         *  @param  trans_mat_array   the 4x4 transformation matrix flattened as std::vector<float>
         *  @return                   the transformed cloud
         */
        PointCloudPtr transformPointCloud(PointCloudPtr input_cloud, std::vector<float> trans_mat_array);

        /*
         *  Helper function to concatenate point clouds
         *
         *  @param  cloud_base  the cloud to which the other cloud will be appended
         *  @param  cloud_add   the cloud to add to the other cloud
        */
        PointCloudPtr addPointClouds(PointCloudPtr cloud_base, PointCloudPtr cloud_add);

        /**
         *  computeCentroid function
         *
         *  This function computes the centroid of a given cloud
         *  @param  input_cloud       the cloud
         *  @return                   the centroid (x,y,z)
         */
        std::vector<float> computeCentroid(PointCloudPtr input_cloud_ptr);


    };

}