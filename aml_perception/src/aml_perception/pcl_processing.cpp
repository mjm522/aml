#include "pcl_processing.h"

namespace aml_pcloud
{

    // ===== PclRosConversions
    PointCloudPtr PclRosConversions::pclCloudFromROSMsg(const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
        PointCloudPtr cloud(new PointCloud);
        PointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*msg, pcl_pc2);
        pcl::fromPCLPointCloud2(pcl_pc2,*cloud);
        return cloud;
    };

    sensor_msgs::PointCloud2::Ptr PclRosConversions::ROSMsgFromPclCloud(PointCloud& cloud)
    {
        sensor_msgs::PointCloud2::Ptr msg(new sensor_msgs::PointCloud2);
        pcl::toROSMsg(cloud, *msg);
        return msg;
    };


    PointCloudPtr PCLProcessor::getCloudFromPcdFile(std::string& input_file)
    {
        PointCloudPtr cloud (new PointCloud);

        if (pcl::io::loadPCDFile<CloudPoint> (input_file, *cloud) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read file %s \n",input_file.c_str());
            return (0);
        }
        else return cloud;
    };

    void PCLProcessor::saveToPcdFile(const std::string filename, const pcl::PCLPointCloud2::Ptr cloud)
    {

        pcl::PCDWriter writer;
        writer.write (filename.c_str(), *cloud, 
            Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), false);
    };

    pcl::PCLPointCloud2::Ptr PCLProcessor::downsamplePcdFile(const pcl::PCLPointCloud2::Ptr cloud)
    {

        pcl::PCLPointCloud2::Ptr cloud_filtered;
        // Create the filtering object
        pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
        sor.setInputCloud (cloud);
        sor.setLeafSize (0.008f, 0.008f, 0.008f);
        sor.filter (*cloud_filtered);

        return cloud_filtered;

    }


    PointCloudPtr PCLProcessor::getPointsNotInPlane(PointCloudPtr input_cloud)
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
    };

    pcl::PointCloud<pcl::PointNormal>::Ptr PCLProcessor::computeNormals(PointCloudPtr cloud)
    {
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals (new pcl::PointCloud<pcl::PointNormal>); // Output datasets
        pcl::IntegralImageNormalEstimation<CloudPoint, pcl::PointNormal> normal_estimator;

        normal_estimator.setNormalEstimationMethod(normal_estimator.AVERAGE_3D_GRADIENT);
        normal_estimator.setMaxDepthChangeFactor(0.02f);
        normal_estimator.setNormalSmoothingSize(10.0f);
        normal_estimator.setInputCloud(cloud);
        normal_estimator.compute(*cloud_normals);

        pcl::copyPointCloud(*cloud, *cloud_normals);

        return cloud_normals;

    };

    PointCloudPtr PCLProcessor::transformPointCloud(PointCloudPtr input_cloud, Eigen::Matrix4f camera_pose)
    {
        /**
         *  Initialize a new point cloud to save the data
         */
        PointCloudPtr new_cloud(new PointCloud);
        pcl::transformPointCloud(*input_cloud, *new_cloud, camera_pose);
        return new_cloud;
    }

    void PCLProcessor::addPointCloud(PointCloudPtr cloud_base, PointCloudPtr cloud_add)
    {
        // get the points from the clouds
        auto &v_base = cloud_base->points;
        auto &v_add  = cloud_add->points;

        // reserve enough space for all clouds
        v_base.reserve(v_base.size() + v_add.size());

        // loop over the points in the cloud to add
        for(const auto &p : v_add) 
        {
            // add the point to the base cloud if the value is not NaN
            if (!std::isnan(p.z)) v_base.emplace_back(p);
        }
    };

    void PCLProcessor::fitPointsToPlane(Eigen::MatrixXf points_mat, Eigen::Vector3f &plane_normal, double &plane_dist) {

        int npts = points_mat.cols(); // number of points = number of columns in matrix; check the size
        
        // first compute the centroid of the data:
        Eigen::Vector3f centroid;
        centroid = Eigen::MatrixXf::Zero(3, 1); // see http://eigen.tuxfamily.org/dox/AsciiQuickReference.txt
        
        //centroid = compute_centroid(points_mat);
         for (int ipt = 0; ipt < npts; ipt++) {
            centroid += points_mat.col(ipt); //add all the column vectors together
        }
        centroid /= npts; //divide by the number of points to get the centroid   

        // subtract this centroid from all points in points_mat:
        Eigen::MatrixXf points_offset_mat = points_mat;
        for (int ipt = 0; ipt < npts; ipt++) {
            points_offset_mat.col(ipt) = points_offset_mat.col(ipt) - centroid;
        }
        //compute the covariance matrix w/rt x,y,z:
        Eigen::Matrix3f CoVar;
        CoVar = points_offset_mat * (points_offset_mat.transpose()); //3xN matrix times Nx3 matrix is 3x3

        // here is a more complex object: a solver for eigenvalues/eigenvectors;
        // we will initialize it with our covariance matrix, which will induce computing eval/evec pairs
        Eigen::EigenSolver<Eigen::Matrix3f> es3f(CoVar);

        Eigen::VectorXf evals; //we'll extract the eigenvalues to here

        // in general, the eigenvalues/eigenvectors can be complex numbers
        //however, since our matrix is self-adjoint (symmetric, positive semi-definite), we expect
        // real-valued evals/evecs;  we'll need to strip off the real parts of the solution

        evals = es3f.eigenvalues().real(); // grab just the real parts
        //cout<<"real parts of evals: "<<evals.transpose()<<endl;

        // our solution should correspond to an e-val of zero, which will be the minimum eval
        //  (all other evals for the covariance matrix will be >0)
        // however, the solution does not order the evals, so we'll have to find the one of interest ourselves

        double min_lambda = evals[0]; //initialize the hunt for min eval
        Eigen::Vector3cf complex_vec; // here is a 3x1 vector of double-precision, complex numbers

        complex_vec = es3f.eigenvectors().col(0); // here's the first e-vec, corresponding to first e-val
        //cout<<"complex_vec: "<<endl;
        //cout<<complex_vec<<endl;
        plane_normal = complex_vec.real(); //strip off the real part
        //cout<<"real part: "<<est_plane_normal.transpose()<<endl;
        //est_plane_normal = es3d.eigenvectors().col(0).real(); // evecs in columns

        double lambda_test;
        int i_normal = 0;
        //loop through "all" ("both", in this 3-D case) the rest of the solns, seeking min e-val
        for (int ivec = 1; ivec < 3; ivec++) {
            lambda_test = evals[ivec];
            if (lambda_test < min_lambda) {
                min_lambda = lambda_test;
                i_normal = ivec; //this index is closer to index of min eval
                plane_normal = es3f.eigenvectors().col(ivec).real();
            }
        }
        // at this point, we have the minimum eval in "min_lambda", and the plane normal
        // (corresponding evec) in "est_plane_normal"/
        // these correspond to the ith entry of i_normal
        plane_dist = plane_normal.dot(centroid);

    };

    void PCLProcessor::fitPointsToPlane(PointCloudPtr input_cloud_ptr, Eigen::Vector3f &plane_normal, double &plane_dist) 
    {
        Eigen::MatrixXf points_mat;
        Eigen::Vector3f cloud_pt;
        //populate points_mat from cloud data;

        int npts = input_cloud_ptr->points.size();
        points_mat.resize(3, npts);

        //somewhat odd notation: getVector3fMap() reading OR WRITING points from/to a pointcloud, with conversions to/from Eigen
        for (int i = 0; i < npts; ++i) {
            cloud_pt = input_cloud_ptr->points[i].getVector3fMap();
            points_mat.col(i) = cloud_pt;
        }
        fitPointsToPlane(points_mat, plane_normal, plane_dist);

    };

    Eigen::Vector3f PCLProcessor::computeCentroid(PointCloudPtr input_cloud_ptr) 
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
        return centroid;
    };

}