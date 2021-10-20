#pragma once

#include <iostream>
#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <vector>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <jsk_recognition_msgs/PolygonArray.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#define NUM_POINTCLOUD_MAXNUM 140000

namespace ptHandle {
    template<typename T>
    sensor_msgs::PointCloud2 cloud2msg(pcl::PointCloud<T> cloud, std::string frame_id = "map")
    {
      sensor_msgs::PointCloud2 cloud_ROS;
      pcl::toROSMsg(cloud, cloud_ROS);
      cloud_ROS.header.frame_id = frame_id;
      return cloud_ROS;
    }

    template<typename T>
    pcl::PointCloud<T> cloudmsg2cloud(sensor_msgs::PointCloud2 cloudmsg){
        pcl::PointCloud<T> cloudresult;
        pcl::fromROSMsg(cloudmsg, cloudresult);
        return cloudresult;
    }

    void pc2pcdfile(const pcl::PointCloud<pcl::PointXYZI>& pc_in, std::string pcd_filename){
        pcl::PointCloud<pcl::PointXYZI> pc_out;
        pc_out = pc_in;
        pc_out.width = pc_out.points.size();
        pc_out.height = 1;
        pcl::io::savePCDFileASCII(pcd_filename, pc_out);

    }

    double xy2theta(const double& x, const double& y){
        if (y >=0){
            return atan2(y,x);
        } else {
            return 2*M_PI + atan2(y,x);
        }
    }

    double xy2radius(const double& x, const double& y){
        return sqrt(pow(x, 2) + pow(y,2));
    }

    float pointDistance(pcl::PointXYZI pt){
        return sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
    } 
    
    bool point_cmp(pcl::PointXYZI a, pcl::PointXYZI b){
        return a.z<b.z;
    }

    cv::Mat ptCloud2cv(pcl::PointCloud<pcl::PointXYZI>& ptCloud){
        //Input : Pointcloud for local frame
        //Output: traversable map for local frame
        cv::Mat ptMat = cv::Mat::zeros(4, ptCloud.points.size(), CV_32FC1);
        
        for (int i = 0; i < (int)ptCloud.points.size(); i++){
            pcl::PointXYZI pt = ptCloud.points[i];
            // if (pt.x > ptRange_max_ || pt.y > ptRange_max_ || pt.z > 0.2) continue;
            
            ptMat.at<float>(0,i) = pt.x;
            ptMat.at<float>(1,i) = pt.y;
            ptMat.at<float>(3,i) = 1;
        }
        return ptMat;
    }
}
