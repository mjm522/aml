
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
#include <pcl/features/pfh.h>

#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud_conversion.h>


namespace aml_pcloud
{
    typedef pcl::PointXYZ CloudPoint; // A point structure denoting Euclidean xyz coordinates
    typedef pcl::PointCloud<CloudPoint> PointCloud;
    typedef pcl::PointCloud<CloudPoint>::Ptr PointCloudPtr;

    typedef pcl::Normal Normal;
    typedef pcl::PointCloud<Normal> NormalCloud;
    typedef pcl::PointCloud<Normal>::Ptr NormalCloudPtr;

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

        // ----- service request.function =  "downsample_cloud"
        PointCloudPtr downsampleCloud(const PointCloudPtr cloud, std::vector<float> &leaf_sizes); 

        // ----- service request.function =  "get_points_not_in_plane"
        PointCloudPtr getPointsNotInPlane(const PointCloudPtr input_cloud); // === CAUSES SEGFAULT !!

        /**
         *  computeNormalForAllPoints function
         *
         *  Computes the normals for all points in a given point cloud. 
         *
         *  
         *  @param  cloud
         *  @return cloud of normals
         */
        // ----- service request.function =  "compute_all_normals"
        PointCloudPtr computeNormalForAllPoints(const PointCloudPtr cloud);
        void computeNormalForAllPoints(const PointCloudPtr cloud, NormalCloudPtr cloud_normals);

        /**
         * fitPlaneAndGetCurvature function
         * Compute the Least-Squares plane fit for a given set of points, using their indices, and return the estimated plane parameters together with the surface curvature.
         * @param cloud
         * @param indices <optional>    the indices of the cloud points to be used for computation
         * 
         * @param plane parameters      the plane parameters as: a, b, c, d (ax + by + cz + d = 0)
         * @param curvature             the estimated surface curvature as a measure of lambda_0/(lambda_0 + lambda_1 + lambda_2)
         */
        // ----- service request.function =  "get_curvature/fit_plane/compute_point_normal"
        void fitPlaneAndGetCurvature(const PointCloudPtr cloud, std::vector< int > indices, std::vector< float > &plane_parameters, float &curvature);
        
        /**
         *  transformPointCloud function
         *
         *  This function takes a point cloud and transforms it according to the transformation defined in trans_mat
         *  @param  input_cloud       the cloud
         *  @param  trans_mat_array   the 4x4 transformation matrix flattened as std::vector<float>
         *  @return                   the transformed cloud
         */
        // ----- service request.function =  "apply_transformation"
        PointCloudPtr transformPointCloud(const PointCloudPtr input_cloud, std::vector<float> trans_mat_array);

        /*
         *  Helper function to concatenate point clouds
         *
         *  @param  cloud_base  the cloud to which the other cloud will be appended
         *  @param  cloud_add   the cloud to add to the other cloud
        */
        // ----- service request.function =  "add_clouds"
        PointCloudPtr addPointClouds(const PointCloudPtr cloud_base, const PointCloudPtr cloud_add);

        /**
         *  computeCentroid function
         *
         *  This function computes the centroid of a given cloud
         *  @param  input_cloud       the cloud
         *  @return                   the centroid (x,y,z)
         */
        // ----- service request.function =  "compute_centroid"
        std::vector<float> computeCentroid(const PointCloudPtr input_cloud_ptr);

        
        /**
         * estimatePfhFeatures function
         * Estimates the Point Feature Histograms for all points in the given cloud (http://pointclouds.org/documentation/tutorials/pfh_estimation.php#pfh-estimation)
         * @param cloud             input cloud containing n points

         * @param pfh_histograms    histogram vector (of float[125] types)
         */
        // ----- Not implemented: should return a vector of type std::vector<float[125]>
        // void estimatePfhFeatures(const PointCloudPtr cloud, std::vector<float> &pfh_histograms);





    private:

        /**
         * Helper Functions to typecast betweeen pcl::PointCloud<pcl::PointXYZ> and pcl::PointCloud<pcl::Normal>.
         * Note: This function does not find normals of the points, but just changes the type for easy conversion to ros messages
         */
        PointCloudPtr normalCloud2PointCloud(const NormalCloudPtr cloud_normals);

        NormalCloudPtr pointCloud2NormalCloud(const PointCloudPtr cloud_points);


    };

}