#include "pcl_processing.h"

namespace aml_pcloud
{

    // ===== PclRosConversions
    PointCloudPtr PclRosConversions::pclCloudFromROSMsg(const sensor_msgs::PointCloud msg)
    {
        // ----- sensor_msgs::PointCloud2 is required for conversions from and to pcl pointclouds
        sensor_msgs::PointCloud2 msg_pc2;
        sensor_msgs::convertPointCloudToPointCloud2(msg, msg_pc2);

        PointCloudPtr cloud(new PointCloud);
        PointCloud2 pcl_pc2;

        pcl_conversions::toPCL(msg_pc2, pcl_pc2);
        pcl::fromPCLPointCloud2(pcl_pc2,*cloud);

        return cloud;
    }

    sensor_msgs::PointCloud PclRosConversions::ROSMsgFromPclCloud(PointCloud& cloud)
    {
        // ----- sensor_msgs::PointCloud2 is required for conversions from and to pcl pointclouds
        sensor_msgs::PointCloud2 msg_pc2;
        sensor_msgs::PointCloud msg;

        pcl::toROSMsg(cloud, msg_pc2);
        // ----- convert back to PointCloud
        sensor_msgs::convertPointCloud2ToPointCloud(msg_pc2, msg);

        return msg;
    }


    PointCloudPtr PCLProcessor::getCloudFromPcdFile(std::string& input_file)
    {
        PointCloudPtr cloud (new PointCloud);

        if (pcl::io::loadPCDFile<CloudPoint> (input_file, *cloud) == -1) //* load the file
        {
            // PCL_ERROR ("Couldn't read file %s \n",input_file.c_str());
            return nullptr;
        }
        else return cloud;
    }

    void PCLProcessor::saveToPcdFile(const std::string filename, const PointCloudPtr cloud)
    {
        pcl::io::savePCDFileASCII (filename, *cloud);
    }

    // may cause seg fault with ROS PCL if using c++11 std
    PointCloudPtr PCLProcessor::downsampleCloud(const PointCloudPtr cloud, std::vector<float> &leaf_sizes)
    {

        // ----- leaf size for downsampling cloud
        if (leaf_sizes.empty())
        {
            for (unsigned i=0; i<3; i++) leaf_sizes.push_back(0.008f);
        }
        else if (leaf_sizes.size() < 3)
        {
            float val = leaf_sizes[0];
            for (unsigned i=0; i<3; i++) leaf_sizes[i] = val;
        }

        pcl::PCLPointCloud2::Ptr cloud_filtered_pc2(new pcl::PCLPointCloud2);
        PointCloudPtr cloud_filtered(new PointCloud);

        pcl::PCLPointCloud2::Ptr cloud_pc2(new pcl::PCLPointCloud2);
        pcl::toPCLPointCloud2(*cloud, *cloud_pc2);

        // ----- Create the filtering object
        pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
        sor.setInputCloud (cloud_pc2);
        sor.setLeafSize (leaf_sizes[0], leaf_sizes[1], leaf_sizes[2]);
        sor.filter (*cloud_filtered_pc2);

        pcl::fromPCLPointCloud2(*cloud_filtered_pc2, *cloud_filtered);

        return cloud_filtered;

    }


    PointCloudPtr PCLProcessor::getPointsNotInPlane(const PointCloudPtr input_cloud)
    {

        PointCloudPtr cloudExtracted(new pcl::PointCloud<pcl::PointXYZ>);

        // Plane segmentation (do not worry, we will see this later).
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::SACSegmentation<pcl::PointXYZ> segmentation;
        segmentation.setOptimizeCoefficients(true);
        segmentation.setModelType(pcl::SACMODEL_PLANE);
        segmentation.setMethodType(pcl::SAC_RANSAC);
        segmentation.setDistanceThreshold(0.01);
        segmentation.setInputCloud(input_cloud);

        // Object for storing the indices.
        pcl::PointIndices::Ptr pointIndices(new pcl::PointIndices);

        segmentation.segment(*pointIndices, *coefficients);

        // Object for extracting points from a list of indices.
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(input_cloud);
        extract.setIndices(pointIndices);
        // We will extract the points that are NOT indexed (the ones that are not in a plane).
        extract.setNegative(true);
        extract.filter(*cloudExtracted);

        return cloudExtracted;
    }
    // END of problematic methods
    void PCLProcessor::fitPlaneAndGetCurvature(const PointCloudPtr cloud, std::vector< int > indices, std::vector< float > &plane_parameters, float &curvature)
    {

        Eigen::Vector4f plane_parameters_eig;

        if (indices.size() == 0)
            pcl::computePointNormal (*cloud, plane_parameters_eig, curvature);
        else pcl::computePointNormal (*cloud, indices, plane_parameters_eig, curvature);
        
        // ----- converting eigen vector to std vector
        std::vector<float> v(plane_parameters_eig.data(), plane_parameters_eig.data() + plane_parameters_eig.rows() * plane_parameters_eig.cols());
        plane_parameters = v;

    }

    PointCloudPtr PCLProcessor::computeNormalForAllPoints(const PointCloudPtr cloud)
    {
        // ---- Output datasets
        NormalCloudPtr cloud_normals (new NormalCloud);

        computeNormalForAllPoints(cloud, cloud_normals);
        // ----- convert to PointCloud type so as to transmit as rosmsg later
        PointCloudPtr out_cloud(new PointCloud);

        out_cloud = normalCloud2PointCloud(cloud_normals);

        return out_cloud;

    }

    void PCLProcessor::computeNormalForAllPoints(const PointCloudPtr cloud, NormalCloudPtr cloud_normals)
    {
        // ----- Create the normal estimation class, and pass the input dataset to it
        pcl::NormalEstimation<CloudPoint, Normal> ne;
        ne.setInputCloud (cloud);

        // ----- Create an empty kdtree representation, and pass it to the normal estimation object.
        // ----- Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
        pcl::search::KdTree<CloudPoint>::Ptr tree (new pcl::search::KdTree<CloudPoint> ());
        ne.setSearchMethod (tree);

        // ----- Use all neighbors in a sphere of radius 3cm
        ne.setRadiusSearch (0.03);

        // ----- Compute the features
        ne.compute (*cloud_normals);


        // ----- cloud_normals->points.size () should have the same size as the input cloud->points.size ()*
    }

    PointCloudPtr PCLProcessor::transformPointCloud(const PointCloudPtr input_cloud, std::vector<float> trans_mat_array)
    {
        /**
         *  Initialize a new point cloud to save the data
         */

        assert (trans_mat_array.size() == 16);

        Eigen::Matrix4f trans_mat_transpose, trans_mat;

        // ----- this fills the matrix in column major style. Transpose is taken since the trans_mat_array is row major.
        for (unsigned i; i < trans_mat_array.size(); i++) // maybe there is a better way of typecasting vector into matrix ?
        {
            trans_mat_transpose(i) = trans_mat_array[i];
        }

        trans_mat = trans_mat_transpose.transpose();

        PointCloudPtr new_cloud(new PointCloud);
        pcl::transformPointCloud(*input_cloud, *new_cloud, trans_mat);

        return new_cloud;
    }

    PointCloudPtr PCLProcessor::addPointClouds(const PointCloudPtr cloud_base, const PointCloudPtr cloud_add)
    {
        // ----- pcl concatenation
        PointCloudPtr cloud_out(new PointCloud);
        *cloud_out = *cloud_base + *cloud_add;
        return cloud_out;
    }

    std::vector<float> PCLProcessor::computeCentroid(const PointCloudPtr input_cloud_ptr) 
    {
        Eigen::Vector3f centroid;
        Eigen::Vector3f cloud_pt;
        int npts = input_cloud_ptr->points.size();
        centroid<<0,0,0;
        //add all the points together:

        for (int ipt = 0; ipt < npts; ipt++) {
            cloud_pt = input_cloud_ptr->points[ipt].getVector3fMap();
            centroid += cloud_pt; //add all the column vectors together
        }
        centroid /= npts; //divide by the number of points to get the centroid

        std::vector<float> centroid_std_vec(centroid.data(), centroid.data() + centroid.rows() * centroid.cols());

        return centroid_std_vec;

    }

    // ===== TODO
    // void PCLProcessor::estimatePfhFeatures(const PointCloudPtr cloud, std::vector<float> &pfh_histograms)
    // {       

    //     NormalCloudPtr normals;
    //     computeNormalForAllPoints(cloud, normals);

    //     // Create the PFH estimation class, and pass the input dataset+normals to it
    //     pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;
    //     pfh.setInputCloud (cloud);
    //     pfh.setInputNormals (normals);
    //     // alternatively, if cloud is of tpe PointNormal, do pfh.setInputNormals (cloud);

    //     // Create an empty kdtree representation, and pass it to the PFH estimation object.
    //     // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    //     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    //     //pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree (new pcl::KdTreeFLANN<pcl::PointXYZ> ()); -- older call for PCL 1.5-
    //     pfh.setSearchMethod (tree);

    //     // Output datasets
    //     pcl::PointCloud<pcl::PFHSignature125>::Ptr pfhs (new pcl::PointCloud<pcl::PFHSignature125> ());

    //     // Use all neighbors in a sphere of radius 5cm
    //     // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
    //     pfh.setRadiusSearch (0.05);

    //     // Compute the features
    //     pfh.compute (*pfhs);

    // }

    PointCloudPtr PCLProcessor::normalCloud2PointCloud(const NormalCloudPtr cloud_normals)
    {
        PointCloudPtr out_cloud(new PointCloud);

        for (unsigned i = 0; i < cloud_normals->size(); i++)
        {
            out_cloud->push_back(CloudPoint(cloud_normals->points[i].normal_x, cloud_normals->points[i].normal_y, cloud_normals->points[i].normal_z));
        }

        return out_cloud;
    }

    NormalCloudPtr PCLProcessor::pointCloud2NormalCloud(const PointCloudPtr cloud_points)
    {
        NormalCloudPtr out_cloud(new NormalCloud);

        for (unsigned i = 0; i < cloud_points->size(); i++)
        {
            out_cloud->push_back(Normal(cloud_points->points[i].x, cloud_points->points[i].y, cloud_points->points[i].z));
        }

        return out_cloud;
    }

}