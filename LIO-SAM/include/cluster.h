#pragma once
#ifndef _CLUSTER_H_
#define _CLUSTER_H_

#include "utility.h"
#include "lshaped_fitting.h"

using namespace std;

enum Type {MEASUREMENT, TRACKER};

class Cluster
{
public:
    int id;
    pcl::PointCloud<PointType> cloud;
    jsk_recognition_msgs::BoundingBox bbox;
    double centroid_x;
    double centroid_y;
    double centroid_z;
    float feature;
    float vel_x;
    float vel_y;
    float m_height;
    float m_area;
    float m_density;
    float m_ratio;

    float m_min_height;
    float m_max_height;
    float m_max_area;
    float m_max_ratio;
    float m_min_density;
    float m_max_density;

    int mode;
    float prob;
    float weight;
    Cluster()
    {
        bbox.value = -1.0;
        mode = -1;
        prob = 0.0;
        weight = 0.0;
        id = -1;
        cloud.reserve(10000);
    }

    Cluster(float min_height, float max_height, float max_area, float max_ratio, float min_density, float max_density)
    : m_min_height(min_height), m_max_height(max_height), m_max_area(max_area), m_max_ratio(max_ratio), m_min_density(min_density), m_max_density(max_density)
    {
        bbox.value = -1.0;
        cloud.reserve(10000); 
        m_height = -1.0;
        m_area = -1.0;
        m_density = -1.0;
        m_ratio = -1.0;
        vel_x = 0.0;
        vel_y = 0.0;
        mode = -1;
        prob = 0.0;
        weight = 0.0;
        id = -1;
    }

    void calculateCentroid()
    {
        centroid_x = 0.0;
        centroid_y = 0.0;
        centroid_z = 0.0;
        const size_t n_points = cloud.points.size();
        for (size_t i = 0u; i <n_points; ++i)
        {
            centroid_x += cloud.points[i].x / n_points;
            centroid_y += cloud.points[i].y / n_points;
            centroid_z += cloud.points[i].z / n_points;
        }      
    }

    void fitBoundingBox()
    {
        // check min max z
        PointType min_pt, max_pt;
        pcl::getMinMax3D(cloud, min_pt, max_pt);
        float height = max_pt.z - min_pt.z;
        printf("%d------%d\n", id, (int)cloud.points.size());
        printf("height: %f\n", height);
        m_height = height;
        if (height < m_min_height || height > m_max_height)
            return;
        
        std::vector<cv::Point2f> hull;
        for (size_t i = 0u; i < cloud.points.size(); ++i)
        {
            hull.push_back(cv::Point2f(cloud.points[i].x, cloud.points[i].y));
        }

        LShapedFIT lshaped;
        TicToc tic_toc;
        cv::RotatedRect rrect = lshaped.FitBox(&hull);
        printf("time: %f\n", tic_toc.toc());
        std::cout << "Shaped-BBox Message : " << rrect.size.width << " " << rrect.size.height << " " << rrect.angle << std::endl;
        
        // check area
        float area = rrect.size.width * rrect.size.height;
        printf("area: %f\n", area);
        m_area = area;
        if (area > m_max_area) 
            return;
        
        // check ratio
        float ratio = rrect.size.height > rrect.size.width ? rrect.size.height / rrect.size.width : rrect.size.width / rrect.size.height;
        printf("ratio: %f\n", ratio);
        m_ratio = ratio;
        if (ratio >= m_max_ratio) {
            // if(rrect.size.height > 1.0 || rrect.size.width > 1.0)
            return;
        }

        float density = cloud.points.size() / (rrect.size.width * rrect.size.height * height);
        printf("density: %f\n", density);
        m_density = density;
        if (density < m_min_density || density > m_max_density) 
            return;
        
        std::vector<cv::Point2f> vertices = lshaped.getRectVertex();
        cv::Point3f center;
        center.z = (max_pt.z + min_pt.z) / 2.0;
        for (size_t i = 0; i < vertices.size(); ++i)
        {
            center.x += vertices[i].x / vertices.size();
            center.y += vertices[i].y / vertices.size();
        }
        bbox.pose.position.x = center.x;
        bbox.pose.position.y = center.y;
        bbox.pose.position.z = center.z;
        bbox.dimensions.x = rrect.size.width;
        bbox.dimensions.y = rrect.size.height;
        bbox.dimensions.z = height;
        bbox.value = 0.0;
        tf::Quaternion quat = tf::createQuaternionFromYaw(rrect.angle * M_PI / 180.0);
        tf::quaternionTFToMsg(quat, bbox.pose.orientation);
    }
};

class ClusterMap
{
private:
    pcl::EuclideanClusterExtraction<PointType> cluster_extractor_;
    std::vector<pcl::PointIndices> cluster_indices_;
    pcl::search::KdTree<PointType>::Ptr kd_tree_;
    pcl::PointCloud<PointType>::Ptr cloud_map_;
    std::vector<Cluster> cluster_map_;
    Type type_;
    double max_r_;
    double max_z_;

    float min_height_;
    float max_height_;
    float max_area_;
    float max_ratio_;
    float min_density_;
    float max_density_;

    std::map<int, int> id_to_idx_;
    
public:
    ClusterMap(){}

    Type getType() {return type_;}
    void setType(Type type) {type_ = type;}

    std::vector<Cluster> getMap(){return cluster_map_;}
    void setMap(std::vector<Cluster> cluster_map) {cluster_map_ = cluster_map;}
    
    std::map<int, int> getIdToIdxMap() {return id_to_idx_;}
    
    std::vector<pcl::PointIndices> getClusterIndices(){return cluster_indices_;}
    
    void clear() { cluster_map_.clear();}

    void init(Type type, double max_r, double max_z, double tol, double min_cluster_size, double max_cluster_size)
    {
        type_ = type;
        max_r_ = max_r;
        max_z_ = max_z;
        setType(type);
        kd_tree_.reset(new pcl::search::KdTree<PointType>());
        cloud_map_.reset(new pcl::PointCloud<PointType>());
        cluster_extractor_.setClusterTolerance(tol);
        cluster_extractor_.setMinClusterSize(min_cluster_size);
        cluster_extractor_.setMaxClusterSize(max_cluster_size);
        cluster_extractor_.setSearchMethod(kd_tree_);
    }

    void setFittingParameters(float min_height, float max_height, float max_area, float max_ratio, float min_density, float max_density)
    {
        min_height_ = min_height;
        max_height_ = max_height;
        max_area_ = max_area;
        max_ratio_ = max_ratio;
        min_density_ = min_density;
        max_density_ = max_density;
    }    
    
    void setFeature(int id, float feature)
    {
        for (std::vector<Cluster>::iterator it = cluster_map_.begin(); it != cluster_map_.end(); ++it)
        {
            if (it->id == id)
            {
                if (it->feature == 0.0 || feature < it->feature)
                    it->feature = feature;
            }
        }
    }

    void buildMap(pcl::PointCloud<PointType>::Ptr cloud_in, int &start_id)
    {
        TicToc tic_toc;

        *cloud_map_ = *cloud_in;

        cluster_indices_.clear();
        cluster_extractor_.setInputCloud(cloud_map_);
        cluster_extractor_.extract(cluster_indices_);
        printf("Points: %d, # of segments: %d\n", cloud_map_->points.size(), (int)cluster_indices_.size());
        ROS_WARN("Clustering: %f ms", tic_toc.toc());
        addClusters(cluster_indices_, start_id);
    }
    
    void addClusters(std::vector<pcl::PointIndices> clusters, int &start_id)
    {
        id_to_idx_.clear();
        TicToc tic_toc;

        for (size_t i = 0u; i < clusters.size(); ++i)
        {
            Cluster cluster(min_height_, max_height_, max_area_, max_ratio_, min_density_, max_density_);
            cluster.id = (int) start_id++;
            id_to_idx_[cluster.id] = i;
            std::vector<int> indices = clusters[i].indices;
            for (size_t j = 0u; j <indices.size(); ++j)
            {
                const size_t index = indices[j];
                PointType point = cloud_map_->points[index];
                point.intensity = cluster.id * 10.0;
                cluster.cloud.points.push_back(point);
            }
            cluster.calculateCentroid();
            cluster.fitBoundingBox();
            cluster_map_.push_back(cluster);
        }
        ROS_WARN("fit bounding box: %f ms", tic_toc.toc());
    }
};

#endif 