#include <pcl/point_cloud.h>  
#include <pcl/point_types.h>  

// #include <pcl/filters/extract_indices.h>
// #include <pcl/segmentation/sac_segmentation.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/filters/random_sample.h>

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

    class PclRosConversions
    {
    public:


        typedef std::shared_ptr<PclRosConversions> Ptr;


        PclRosConversions() {}

    /**
     * Helper function to convert from sensor_msgs/PointCloud2 to pcl::PointCloud<pcl::PointXYZRGB>
     */
        PointCloudPtr pclCloudFromROSMsg(const sensor_msgs::PointCloud2 msg);

    /**
     *----- Helper function to convert from pcl::PointCloud<pcl::PointXYZ> to sensor_msgs/PointCloud2
     */
        sensor_msgs::PointCloud2::Ptr ROSMsgFromPclCloud(PointCloud& cloud);
  

    };

    class PCLProcessor
    {

    public:

        typedef std::shared_ptr<PCLProcessor> Ptr;

        PCLProcessor() {}

        // ----- service request.function =  "read_pcd_file"
        PointCloudPtr getCloudFromPcdFile(std::string& input_file);

        // ----- service request.function =  "save_to_file"
        void saveToPcdFile(std::string filename, const PointCloudPtr cloud);

        // pcl::PCLPointCloud2::Ptr downsamplePcdFile(const pcl::PCLPointCloud2::Ptr cloud); // === CAUSES SEGFAULT !!

        // PointCloudPtr getPointsNotInPlane(PointCloudPtr input_cloud); // === CAUSES SEGFAULT !!

        /**
         *  computeNormals function
         *
         *  Computes the normals for a given point cloud. Taken from the example from blackboard and
         *  edited to fit in our program
         *
         *  @see http://blackboard.uva.nl/ -> 20152016 CV2 
         *  
         *  @param  cloud
         *  @return cloud with normals
         */
        pcl::PointCloud<pcl::PointNormal>::Ptr computeNormals(PointCloudPtr cloud);

        /**
         *  TransformPointCloud function
         *
         *  This function takes a point cloud and transforms it according to the camera pose
         *  @param  normal_cloud the cloud with normals
         *  @param  camera_pose  the camera pose
         *  @return              the cloud transformed
         */
        PointCloudPtr transformPointCloud(PointCloudPtr input_cloud, Eigen::Matrix4f camera_pose);

        /*
         *  Helper function to concatenate point clouds, with help from TA via Github
         *  @see https://github.com/Tomaat/CV2/issues/1
         *
         *  @param  cloud_base  the cloud to which the other cloud will be appended
         *  @param  cloud_add   the cloud to add to the other cloud
        */
        void addPointCloud(PointCloudPtr cloud_base, PointCloudPtr cloud_add);

        /**provide an array of 3-D points (in columns), and this function will use and eigen-vector approach to find the best-fit plane
         * It returns the plane's normal vector and the plane's (signed) distance from the origin.
         * @param points_array input: points_array is a matrix of 3-D points to be plane-fitted; coordinates are in columns
         * @param plane_normal output: this function will compute components of the plane normal here
         * @param plane_dist output: scalar (signed) distance of the plane from the origin
         */
        
        void fitPointsToPlane(Eigen::MatrixXf points_array, 
                                Eigen::Vector3f &plane_normal, 
                                double &plane_dist); 
        void fitPointsToPlane(PointCloudPtr input_cloud_ptr,Eigen::Vector3f &plane_normal, double &plane_dist);

        Eigen::Vector3f computeCentroid(PointCloudPtr input_cloud_ptr);


    };

}