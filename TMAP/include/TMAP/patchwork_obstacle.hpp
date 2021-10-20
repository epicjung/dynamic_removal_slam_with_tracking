#ifndef PATCHWORK_M_H
#define PATCHWORK_M_H

#include <iostream>
#include <Eigen/Dense>
#include <vector>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/common/centroid.h>

#include <jsk_recognition_msgs/PolygonArray.h>
#include <sensor_msgs/PointCloud2.h>

#include "../../utils/ptcloud_handle.hpp"


#define SURFACE_ENOUGH 0.2
#define NORMAL_ENOUGH 0.55 //MH Question
#define Z_THR_FILTERED 0.0
#define TOO_TILTED 1.0

#define COV_CONSTRAINT 0.25
#define NUM_EXPERINEMTAL_MAX_PATCH 10000

#define NEIGHBOR_RANGE 12.3625
#define QUATER_RANGE 22.025
#define MID_RANGE 41.35
#define NUM_DISKS 4

using Eigen::MatrixXf;
using Eigen::JacobiSVD;
using Eigen::VectorXf;

typedef std::vector<pcl::PointCloud<pcl::PointXYZI>> Ring;
typedef std::vector<Ring> RegionwisePatches;

class Patchwork_M{
public:
    Patchwork_M(){};
    Patchwork_M(ros::NodeHandle* nh):node_handle_(*nh){
        // Init ROS related
        ROS_INFO("Inititalizing Patchwork_M...");

        node_handle_.param("sensor_height", sensor_height_, 1.723);
        ROS_INFO("Sensor Height: %f", sensor_height_);
        node_handle_.param<std::string>("/patchwork/mode", mode_, "non_uniform"); // whether non_uniform or uniform
        node_handle_.param<bool>("/patchwork/use_z_thr", use_z_thr_, false);
        node_handle_.param<bool>("/patchwork/tuning_mode", tuning_mode_, false);
        node_handle_.param<bool>("/patchwork/reject_reflection_error",reject_reflection_error_, false);

        std::cout<<"\033[1;32m MODE: "<<mode_<<" use z thr.: "<<use_z_thr_<<" tuning_mode: "<<tuning_mode_<<"\033[0m"<<std::endl;
        std::cout<<"\033[1;32m Reject Reflection Error: "<< reject_reflection_error_<< "\033[0m"<<std::endl;
        
        node_handle_.param("/patchwork/num_iter", num_iter_, 3);
        ROS_INFO("Num of Iteration: %d", num_iter_);

        node_handle_.param("/patchwork/num_lpr", num_lpr_, 20);
        ROS_INFO("Num of LPR: %d", num_lpr_);

        node_handle_.param("/patchwork/num_min_pts", num_min_pts_, 10);
        ROS_INFO("Num of min. points: %d", num_min_pts_);

        node_handle_.param("/patchwork/th_seeds", th_seeds_, 0.4);
        ROS_INFO("Seeds Threshold: %f", th_seeds_);

        node_handle_.param("/patchwork/th_dist", th_dist_, 0.3);
        ROS_INFO("Distance Threshold: %f", th_dist_);

        node_handle_.param("/patchwork/max_r", max_range_, 80.0);
        ROS_INFO("Max. range:: %f", max_range_);

        node_handle_.param("/patchwork/min_r", min_range_, 2.7); // It indicates bodysize of the car.
        ROS_INFO("Min. range:: %f", min_range_);

        node_handle_.param("/patchwork/num_rings", num_rings_, 30);
        ROS_INFO("Num. rings: %d", num_rings_);

        node_handle_.param("/patchwork/num_sectors", num_sectors_, 108);
        ROS_INFO("Num. sectors: %d", num_sectors_);

        node_handle_.param("/patchwork/normal_thr", normal_thr_, 0.5); // The more larger, the more strict
        ROS_INFO("Normal vector threshold: %f", normal_thr_);

        node_handle_.param("/patchwork/obstacle_height_thr", obstacle_height_thr_, 2.0); // The more larger, the more strict
        ROS_INFO("Normal vector threshold: %f", obstacle_height_thr_);

        node_handle_.param("/patchwork/adaptive_seed_selection_margin", adaptive_seed_selection_margin_, -1.1); // The more larger, the more soft
        ROS_INFO("adaptive_seed_selection_margin:");
        std::cout<<adaptive_seed_selection_margin_<<std::endl;


        node_handle_.getParam("/patchwork/Z_THR_", Z_THR_);
        node_handle_.getParam("/patchwork/SV_THR_", SV_THR_);
        std::cout << "Z_THR_: "<< Z_THR_[0]<< ", "<< Z_THR_[1]<< ", "<< Z_THR_[2]<< ", "<< Z_THR_[3] << std::endl;
        std::cout << "SV_THR_: "<< SV_THR_[0]<< ", "<< SV_THR_[1]<< ", "<< SV_THR_[2]<< ", "<< SV_THR_[3]<< std::endl;
        node_handle_.param("/patchwork/visualize", visualize_, true);
        node_handle_.param<std::string>("/seq", seq_, "00");
        poly_list_.header.frame_id = "/base_link";
        poly_list_.polygons.reserve(132000);

        // Used when uniform
        ring_size = (max_range_ - min_range_) / num_rings_;
        sector_size = 2 * M_PI / num_sectors_;
        revert_pc.reserve(NUM_EXPERINEMTAL_MAX_PATCH);
        ground_pc_.reserve(NUM_EXPERINEMTAL_MAX_PATCH);
        non_ground_pc_.reserve(NUM_EXPERINEMTAL_MAX_PATCH);
        obstacle_pc_.reserve(NUM_EXPERINEMTAL_MAX_PATCH);
        patchwise_ground_.reserve(NUM_EXPERINEMTAL_MAX_PATCH);
        patchwise_nonground_.reserve(NUM_EXPERINEMTAL_MAX_PATCH);
        patchwise_obstacle_.reserve(NUM_EXPERINEMTAL_MAX_PATCH);

        PlaneViz = node_handle_.advertise<jsk_recognition_msgs::PolygonArray>("/patchwork/patches",100);
        revert_pc_pub = node_handle_.advertise<sensor_msgs::PointCloud2>("/revert_pc",100);
        reject_pc_pub = node_handle_.advertise<sensor_msgs::PointCloud2>("/reject_pc",100);

        if (mode_ == "uniform") init_regionwise_patches(patches_, num_sectors_, num_rings_);
        else if (mode_ == "non_uniform"){
            num_sectors_list_ = {16, 32, 54, 32};
            num_rings_list_ = {2, 4, 4, 4};
            num_lprs_set_= {20, 20, 20, 20};
            min_ranges = {min_range_, NEIGHBOR_RANGE, QUATER_RANGE, MID_RANGE};
            ring_sizes = {(NEIGHBOR_RANGE - min_range_) / num_rings_list_.at(0), (QUATER_RANGE-NEIGHBOR_RANGE) / num_rings_list_.at(1), (MID_RANGE - QUATER_RANGE) / num_rings_list_.at(2), (max_range_ - MID_RANGE) / num_rings_list_.at(3)};
            sector_sizes = {2 * M_PI / num_sectors_list_.at(0), 2 * M_PI / num_sectors_list_.at(1), 2 * M_PI / num_sectors_list_.at(2), 2 * M_PI / num_sectors_list_.at(3)};
            std::cout<<"INITIALIZATION COMPLETE"<<std::endl;

            for (int iter = 0; iter < NUM_DISKS;++iter){
                RegionwisePatches patches;
                init_regionwise_patches(patches, num_sectors_list_.at(iter), num_rings_list_.at(iter));
                tripatches_.push_back(patches);
            }
        }
    }

    void estimate_ground(const pcl::PointCloud<pcl::PointXYZI>& cloudIn,
                         pcl::PointCloud<pcl::PointXYZI>& cloudOut,
                         pcl::PointCloud<pcl::PointXYZI>& cloudNonground,
                         double& time_taken);

    //viz
    geometry_msgs::PolygonStamped set_plane_polygon(const MatrixXf& normal_v, const float& d);
    pcl::PointCloud<pcl::PointXYZI> get_obstacle_pc(){
      return obstacle_pc_;
    }

private:
    ros::NodeHandle node_handle_;

    // ROS parameters
    double sensor_height_;
    int num_iter_;
    int num_lpr_;
    int num_min_pts_;
    double th_seeds_;
    double th_dist_;
    double max_range_;
    double min_range_;
    int num_rings_;
    int num_sectors_;
    double normal_thr_;
    double adaptive_seed_selection_margin_;
    std::string mode_;
    std::string seq_;
    bool use_z_thr_;
    bool tuning_mode_;
    bool reject_reflection_error_;

    float d_;
    MatrixXf normal_;
    VectorXf singular_values_;
    float th_dist_d_;
    Eigen::Matrix3f cov_;
    Eigen::Vector4f pc_mean_;
    double ring_size;
    double sector_size;
    double obstacle_height_thr_;
    double th_dist_obstacle_d_;

    // For visualization
    bool visualize_;

    std::vector<int> num_sectors_list_, num_rings_list_, num_lprs_set_;
    std::vector<double> sector_sizes, ring_sizes, min_ranges, th_seeds_set_;
    std::vector<RegionwisePatches> tripatches_;
    jsk_recognition_msgs::PolygonArray poly_list_;

    int RING_BOUNDARY = 4;

    std::vector<double> Z_THR_;
    std::vector<double> SV_THR_;

    RegionwisePatches patches_;

    ros::Publisher PlaneViz, revert_pc_pub, reject_pc_pub;
    pcl::PointCloud<pcl::PointXYZI> revert_pc , reject_pc;
    pcl::PointCloud<pcl::PointXYZI> ground_pc_;
    pcl::PointCloud<pcl::PointXYZI> non_ground_pc_;
    pcl::PointCloud<pcl::PointXYZI> obstacle_pc_;

    pcl::PointCloud<pcl::PointXYZI> patchwise_ground_;
    pcl::PointCloud<pcl::PointXYZI> patchwise_nonground_;
    pcl::PointCloud<pcl::PointXYZI> patchwise_obstacle_;
    //
    void init_regionwise_patches(RegionwisePatches& patches, int num_sectors, int num_rings);
    void clear_patches(RegionwisePatches& patches, int num_sectors, int num_rings);
    void pc2patches(const pcl::PointCloud<pcl::PointXYZI>& src, RegionwisePatches& patches);
    void pc2tripatches(const pcl::PointCloud<pcl::PointXYZI>& src, std::vector<RegionwisePatches>& tripatches);

    void extract_patchwiseground(const pcl::PointCloud<pcl::PointXYZI>& src,
                                 pcl::PointCloud<pcl::PointXYZI>& dst,
                                 pcl::PointCloud<pcl::PointXYZI>& non_ground_dst);
    void extract_patchwiseground(const pcl::PointCloud<pcl::PointXYZI>& src,
                                 pcl::PointCloud<pcl::PointXYZI>& dst,
                                 pcl::PointCloud<pcl::PointXYZI>& non_ground_dst,
                                 pcl::PointCloud<pcl::PointXYZI>& obstacle_dst);                                 
    void estimate_plane_(const pcl::PointCloud<pcl::PointXYZI>& ground);
    void extract_initial_seeds_(const pcl::PointCloud<pcl::PointXYZI>& p_sorted,
                                      pcl::PointCloud<pcl::PointXYZI>& init_seeds);

    void extract_patchwiseground(const int zone_idx, const pcl::PointCloud<pcl::PointXYZI>& src,
                                 pcl::PointCloud<pcl::PointXYZI>& dst,
                                 pcl::PointCloud<pcl::PointXYZI>& non_ground_dst);
    void extract_patchwiseground(const int zone_idx, const pcl::PointCloud<pcl::PointXYZI>& src,
                                 pcl::PointCloud<pcl::PointXYZI>& dst,
                                 pcl::PointCloud<pcl::PointXYZI>& non_ground_dst,
                                 pcl::PointCloud<pcl::PointXYZI>& obstacle_dst);                                    
    void extract_initial_seeds_(const int zone_idx, const pcl::PointCloud<pcl::PointXYZI>& p_sorted,
                                      pcl::PointCloud<pcl::PointXYZI>& init_seeds);

    geometry_msgs::PolygonStamped set_polygons(int r_idx, int theta_idx, int num_split);
    geometry_msgs::PolygonStamped set_polygons(int zone_idx, int r_idx, int theta_idx, int num_split);

    float calc_pt_z(float pt_x, float pt_y);
};

void Patchwork_M::init_regionwise_patches(RegionwisePatches& patches, int num_sectors, int num_rings){
  pcl::PointCloud<pcl::PointXYZI> cloud;
  cloud.reserve(1000);
  Ring ring;
  for (int i=0; i< num_sectors; i++){
    ring.emplace_back(cloud);
  }
  for (int j=0; j< num_rings; j++){
    patches.emplace_back(ring);
  }
}

void Patchwork_M::clear_patches(RegionwisePatches& patches, int num_sectors, int num_rings){
  for (int i=0; i<num_sectors; i++){
    for (int j=0; j<num_rings; j++){
      if(!patches[j][i].points.empty()) patches[j][i].points.clear();
    }
  }
}

void Patchwork_M::estimate_ground(const pcl::PointCloud<pcl::PointXYZI>& cloudIn,
                                pcl::PointCloud<pcl::PointXYZI>& cloudOut,
                                pcl::PointCloud<pcl::PointXYZI>& cloudNonground,
                                double& time_taken){
  static time_t start, end;
  pcl::PointCloud<pcl::PointXYZI> cloudObstacle;
  // 0. Clear  
  poly_list_.header.stamp = ros::Time::now();
  if (!poly_list_.polygons.empty()) poly_list_.polygons.clear();
  if (!poly_list_.likelihood.empty()) poly_list_.likelihood.clear();
  pcl::PointCloud<pcl::PointXYZI> ptCloudIn = cloudIn;
  std::string cloud_frame = cloudIn.header.frame_id;

  start = clock();  

  // Pre-processing
  // 1. Sort points in cloud on z-axis value.
  sort(ptCloudIn.points.begin(), ptCloudIn.end(), ptHandle::point_cmp);

  // 2. Error point removal
  // As there are some error mirror reflection under the ground, 
  // here regardless point under 2* sensor height
  if (reject_reflection_error_){
    pcl::PointCloud<pcl::PointXYZI>::iterator it = ptCloudIn.points.begin();
    for (int i=0; i<ptCloudIn.points.size();i++){
      if(ptCloudIn.points[i].z < -adaptive_seed_selection_margin_*sensor_height_){
        it ++;
      } else {    
        break;
      }
    }
    ptCloudIn.points.erase(ptCloudIn.points.begin(),it);
  }

  // Patchwork_M
  // 3. pointcloud -> set patchwise-pointcloud
  if (mode_== "uniform"){
    clear_patches(patches_, num_sectors_, num_rings_);
    pc2patches(ptCloudIn, patches_);

  }
  else if (mode_=="non_uniform"){
    for(int k=0; k<NUM_DISKS; k++){
      // std::cout << k << ", " <<num_sectors_list_[k]<<", "<<num_rings_list_[k] <<std::endl;
      clear_patches(tripatches_[k], num_sectors_list_[k], num_rings_list_[k]);
    }
    pc2tripatches(ptCloudIn, tripatches_);
  }

  // 4. patchwise ground segmentation

  cloudOut.clear();
  cloudNonground.clear();
  cloudObstacle.clear();
  obstacle_pc_.clear();
  if(!revert_pc.empty()) revert_pc.clear();
  if(!reject_pc.empty()) reject_pc.clear();

  if (mode_ == "uniform"){  // 4(a) uniform-sized patchwise ground extraction
    for (uint16_t ring_idx = 0; ring_idx < num_rings_; ++ring_idx){
      for (uint16_t sector_idx = 0; sector_idx < num_sectors_; ++sector_idx){

        if(patches_[ring_idx][sector_idx].points.size() > num_min_pts_){
          // extract_patchwiseground(patches_[ring_idx][sector_idx], patchwise_ground_, patchwise_nonground_);
          extract_patchwiseground(patches_[ring_idx][sector_idx], patchwise_ground_, patchwise_nonground_, patchwise_obstacle_);

          if(visualize_){
            geometry_msgs::PolygonStamped polygons = set_polygons(ring_idx, sector_idx, 3);
            polygons.header = poly_list_.header;
            poly_list_.polygons.push_back(polygons);

            if(normal_(2,0) > normal_thr_){ 
              // orthogonal
              if (ring_idx < RING_BOUNDARY){
                if(pc_mean_(2,0) > Z_THR_[ring_idx]){
                  poly_list_.likelihood.push_back(Z_THR_FILTERED);
                } else{
                  poly_list_.likelihood.push_back(NORMAL_ENOUGH);
                } 
              }else{
                poly_list_.likelihood.push_back(NORMAL_ENOUGH);
              }
            } else{ 
              // tilted
              poly_list_.likelihood.push_back(TOO_TILTED);
            }
          }

          if (normal_(2,0) < normal_thr_){
            cloudNonground += patchwise_ground_;
            cloudNonground += patchwise_nonground_;
            cloudObstacle += patchwise_obstacle_;
          } else{
            cloudOut += patchwise_ground_;
            cloudNonground += patchwise_nonground_;
            cloudObstacle += patchwise_obstacle_;
          }
        }
      }
    }
  }
  else if (mode_=="non_uniform"){ // 4(b) non-uniformed patchwise ground extraction
    for (int k=0; k<NUM_DISKS; ++k){
      auto patches = tripatches_[k];
      for (uint16_t ring_idx=0; ring_idx<num_rings_list_[k]; ++ring_idx){
        for (uint16_t sector_idx=0; sector_idx<num_sectors_list_[k]; ++sector_idx){
          if(patches[ring_idx][sector_idx].points.size() > num_min_pts_){
            extract_patchwiseground(k, patches[ring_idx][sector_idx], patchwise_ground_, patchwise_nonground_, patchwise_obstacle_);

            double surface_variable = singular_values_.minCoeff()/(singular_values_(0) + singular_values_(1) + singular_values_(2));

            if (visualize_){
              geometry_msgs::PolygonStamped polygons = set_polygons(k, ring_idx, sector_idx, 3);
              polygons.header = poly_list_.header;
              poly_list_.polygons.push_back(polygons);

              if (abs(normal_(2,0)) > normal_thr_){ //orthogonal
                if ( (k < 2) && (ring_idx < 2) && use_z_thr_){
                    if (pc_mean_(2,0) > Z_THR_[ring_idx + 2 * k]){
                      if (SV_THR_[ring_idx + 2 * k] > surface_variable){
                        poly_list_.likelihood.push_back(SURFACE_ENOUGH);
                      }else{
                        poly_list_.likelihood.push_back(Z_THR_FILTERED);
                      }

                    }else{
                      poly_list_.likelihood.push_back(NORMAL_ENOUGH);
                    }
                }else{
                  poly_list_.likelihood.push_back(NORMAL_ENOUGH);
                }
              }else{ // tilted
                poly_list_.likelihood.push_back(TOO_TILTED);
              }
            }
            if (abs(normal_(2,0))<normal_thr_){
              cloudNonground += patchwise_ground_;
              cloudNonground += patchwise_nonground_;
              cloudObstacle += patchwise_obstacle_;
            } else{
              // satisfy orthogonality
              if ( (k<2) && (ring_idx < 2) && use_z_thr_){
                if (pc_mean_(2,0) >Z_THR_[ring_idx + 2*k]){

                  if (SV_THR_[ring_idx + 2*k] > surface_variable){
                    if (tuning_mode_){
                      // std::cout<<"\033[1;36m REVERT operated. Check "<<ring_idx + 2 * k<<"th parameter SV_THR_: "<<SV_THR_[ring_idx + 2 * k]<<" > "<<surface_variable<<"\033[0m"<<std::endl;
                      revert_pc += patchwise_ground_;
                    }
                    cloudOut += patchwise_ground_;
                    cloudNonground += patchwise_nonground_;
                    cloudObstacle += patchwise_obstacle_;
                  } else{
                    if (tuning_mode_){
                      // std::cout<<"\033[1;35m REJECTION operated. Check "<<ring_idx + 2 * k<<"th paramete of Z_THR_: "<<Z_THR_[ring_idx + 2 * k]<<" < "<<pc_mean_(2,0)<<"\033[0m"<<std::endl;
                      reject_pc += patchwise_ground_;
                    }
                    cloudNonground += patchwise_nonground_;
                    cloudNonground += patchwise_nonground_;
                    cloudObstacle += patchwise_obstacle_;
                  }

                } else{
                  cloudOut += patchwise_ground_;
                  cloudNonground += patchwise_nonground_; 
                  cloudObstacle += patchwise_obstacle_;
                }

              } else{
                cloudOut += patchwise_ground_;
                cloudNonground += patchwise_nonground_;
                cloudObstacle += patchwise_obstacle_;
              }
            }
          }
        }
      }
    }
    if (tuning_mode_){
      sensor_msgs::PointCloud2 cloud_ROS;
      pcl::toROSMsg(revert_pc, cloud_ROS);
      cloud_ROS.header.stamp = ros::Time::now();
      cloud_ROS.header.frame_id = cloud_frame;
      revert_pc_pub.publish(cloud_ROS);
      pcl::toROSMsg(reject_pc, cloud_ROS);
      cloud_ROS.header.stamp = ros::Time::now();
      cloud_ROS.header.frame_id = cloud_frame;
      reject_pc_pub.publish(cloud_ROS);
    }
  }
  obstacle_pc_ = cloudObstacle;
  end = clock();
  time_taken = (double)(end-start)/CLOCKS_PER_SEC;
  PlaneViz.publish(poly_list_);
}

void Patchwork_M::pc2patches(const pcl::PointCloud<pcl::PointXYZI>& src, RegionwisePatches& patches){
  for(auto const &pt : src.points){
    int ring_idx, sector_idx;

    double r = ptHandle::xy2radius(pt.x, pt.y);
    if ( (r <=max_range_) && (r >= min_range_) ){
      double theta = ptHandle::xy2theta(pt.x, pt.y);
      
      ring_idx   = std::min(static_cast<int>(((r - min_range_)/ring_size)), num_rings_-1   );
      sector_idx = std::min(static_cast<int>((theta/sector_size))         , num_sectors_-1 );

      patches[ring_idx][sector_idx].points.push_back(pt);
    }
  }
}

void Patchwork_M::pc2tripatches(const pcl::PointCloud<pcl::PointXYZI>& src, std::vector<RegionwisePatches>& tripatches){
  for(auto const &pt : src.points){
    int ring_idx, sector_idx;

    double r = ptHandle::xy2radius(pt.x, pt.y);
    if ( (r<=max_range_) && (r>=min_range_)){
      double theta = ptHandle::xy2theta(pt.x, pt.y);
      if (theta < 0.0)
        continue;
      if (r<min_ranges[1]){
        ring_idx   = std::min(static_cast<int>((r-min_ranges[0])/ring_sizes[0]) ,num_rings_list_[0]  -1 ); 
        sector_idx = std::min(static_cast<int>( theta/sector_sizes[0])          ,num_sectors_list_[0]-1 );
        tripatches[0][ring_idx][sector_idx].points.emplace_back(pt);
      }
      else if (r<min_ranges[2]){
        ring_idx   = std::min(static_cast<int>((r-min_ranges[1])/ring_sizes[1]) ,num_rings_list_[1]  -1 );
        sector_idx = std::min(static_cast<int>( theta/sector_sizes[1])          ,num_sectors_list_[1]-1 );
        tripatches[1][ring_idx][sector_idx].points.emplace_back(pt);
      }
      else if (r<min_ranges[3]){
        ring_idx   = std::min(static_cast<int>((r-min_ranges[2])/ring_sizes[2]) , num_rings_list_[2]  -1);
        sector_idx = std::min(static_cast<int>( theta/sector_sizes[2])          , num_sectors_list_[2]-1);
        tripatches[2][ring_idx][sector_idx].points.emplace_back(pt);
      }
      else{
        ring_idx   = std::min(static_cast<int>((r-min_ranges[3])/ring_sizes[3]) , num_rings_list_[3]  -1);
        sector_idx = std::min(static_cast<int>( theta/sector_sizes[3])          , num_sectors_list_[3]-1);
        tripatches[3][ring_idx][sector_idx].points.emplace_back(pt);
      }
    }

  }
  
}

float Patchwork_M::calc_pt_z(float pt_x, float pt_y){
  float& a_ = normal_(0,0);
  float& b_ = normal_(1,0);
  float& c_ = normal_(2,0);
  return -2.2;
//  return (d_ - (a_ * pt_x + b_ * pt_y) ) / c_;
}

void Patchwork_M::estimate_plane_(const pcl::PointCloud<pcl::PointXYZI>& ground){
  // function for uniform mode
  pcl::computeMeanAndCovarianceMatrix(ground, cov_, pc_mean_);
  
  // Singular Value Decomposition: SVD
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov_, Eigen::DecompositionOptions::ComputeFullU);
  singular_values_ = svd.singularValues();
  // std::cout<<"\033[1;32mSingular values are ";
  // std::cout<<singular_values_<<"\033[0m"<<std::endl;
  
  // Use the least singular vector as normal
  normal_ = (svd.matrixU().col(2));

  // mean ground seeds value
  Eigen::Vector3f seeds_mean = pc_mean_.head<3>();

  // according to noraml.T*[x,y,z] = -d
  d_ = -(normal_.transpose()*seeds_mean)(0,0);

  // set distance theshold to 'th_dist - d'
  th_dist_d_ = th_dist_ - d_;
  th_dist_obstacle_d_ = obstacle_height_thr_ - d_;
}

void Patchwork_M::extract_initial_seeds_(const pcl::PointCloud<pcl::PointXYZI>& p_sorted,
                                  pcl::PointCloud<pcl::PointXYZI>& init_seeds){
  //function for uniform mode
  init_seeds.points.clear();

  // LPR is the mean of Low Point Representative
  double sum = 0;
  int cnt = 0;

  // Calculate the mean height value.

  for (int i=0; i<p_sorted.points.size() && cnt<num_lpr_; i++){
    sum += p_sorted.points[i].z;
    cnt++;
  }

  double lpr_height = cnt!=0?sum/cnt:0;

  for(int i=0 ; i<p_sorted.points.size() ; i++){
    if(p_sorted.points[i].z < lpr_height + th_seeds_){
      init_seeds.points.push_back(p_sorted.points[i]);
    }
  }
}

void Patchwork_M::extract_patchwiseground(const pcl::PointCloud<pcl::PointXYZI>& src,
                                        pcl::PointCloud<pcl::PointXYZI>& dst,
                                        pcl::PointCloud<pcl::PointXYZI>& non_ground_dst){
  //function for uniform mode
  // 0. Initialization
  if (!ground_pc_.empty()) ground_pc_.clear();  
  if (!dst.empty()) dst.clear();
  if (!non_ground_dst.empty()) non_ground_dst.clear();

  // 1. Set seeds
  extract_initial_seeds_(src, ground_pc_);

  // 2. Extract ground
  for(int i=0; i<num_iter_; i++){
    estimate_plane_(ground_pc_);
    ground_pc_.clear();

    // pointcloud to matrix (Nx3 Matrix)
    Eigen::MatrixXf points(src.points.size(),3);
    int j = 0;
    for (auto& p:src.points){
      points.row(j++)<<p.x,p.y,p.z;
    }

    // ground plane model (Nx3)*(3x1) = Nx1
    Eigen::VectorXf result = points*normal_;

    // threshold filter
    for (int r=0; r<result.rows(); r++){
      if (i<num_iter_-1){
        if(result[r]<th_dist_d_){
          ground_pc_.points.push_back(src[r]);
        }
      } else{
        // Final iteration
        if(result[r]<th_dist_d_){
          dst.points.push_back(src[r]);
        } else{
          if (i==num_iter_-1){
            non_ground_dst.push_back(src[r]);
          }
        }
      }
    }

  }
}

void Patchwork_M::extract_patchwiseground(const pcl::PointCloud<pcl::PointXYZI>& src,
                                        pcl::PointCloud<pcl::PointXYZI>& dst,
                                        pcl::PointCloud<pcl::PointXYZI>& non_ground_dst,
                                        pcl::PointCloud<pcl::PointXYZI>& obstacle_dst){
  //function for uniform mode
  // 0. Initialization
  if (!ground_pc_.empty()) ground_pc_.clear();  
  if (!dst.empty()) dst.clear();
  if (!non_ground_dst.empty()) non_ground_dst.clear();
  if (!obstacle_dst.empty()) obstacle_dst.clear();

  // 1. Set seeds
  extract_initial_seeds_(src, ground_pc_);

  // 2. Extract ground
  for(int i=0; i<num_iter_; i++){
    estimate_plane_(ground_pc_);
    ground_pc_.clear();

    // pointcloud to matrix (Nx3 Matrix)
    Eigen::MatrixXf points(src.points.size(),3);
    int j = 0;
    for (auto& p:src.points){
      points.row(j++)<<p.x,p.y,p.z;
    }

    // ground plane model (Nx3)*(3x1) = Nx1
    Eigen::VectorXf result = points*normal_;

    // threshold filter
    for (int r=0; r<result.rows(); r++){
      if (i<num_iter_-1){
        if(result[r]<th_dist_d_){
          ground_pc_.points.push_back(src[r]);
        }
      } else{
        // Final iteration
        if(result[r]<th_dist_d_){
          dst.points.push_back(src[r]);
        } else{
          if (i==num_iter_-1){
            if(result[r]<th_dist_obstacle_d_){
              obstacle_dst.push_back(src[r]);
            }
            non_ground_dst.push_back(src[r]);
          }
        }
      }
    }
  }
}

geometry_msgs::PolygonStamped Patchwork_M::set_polygons(int r_idx, int theta_idx, int num_split){
  //function for uniform mode
  //num_split is defined to make curved polygon.

  geometry_msgs::PolygonStamped polygons;

  // set corner point of polygon. Start from Right Low and CCW
  geometry_msgs::Point32 corner;

  //Right-Lower
  double r_len = r_idx * ring_size + min_range_;
  double angle = theta_idx * sector_size;

  corner.x = r_len * cos(angle); corner.y = r_len * sin(angle);
  corner.z = calc_pt_z(corner.x, corner.y);
  polygons.polygon.points.push_back(corner);
  
  //Right-Upper
  r_len = r_len + ring_size;
  corner.x = r_len * cos(angle); corner.y = r_len * sin(angle);
  corner.z = calc_pt_z(corner.x, corner.y);
  polygons.polygon.points.push_back(corner);

  // Right-Upper -> Left-Upper
  for (int idx = 1; idx <= num_split; ++idx){
    angle = angle + sector_size/num_split;
    corner.x = r_len*cos(angle); corner.y = r_len*sin(angle);
    corner.z = calc_pt_z(corner.x, corner.y);
    polygons.polygon.points.push_back(corner);
  }

  //Left-Lower
  r_len = r_len - ring_size;
  corner.x = r_len * cos(angle); corner.y = r_len * sin(angle);
  corner.z = calc_pt_z(corner.x, corner.y);
  polygons.polygon.points.push_back(corner);

  //Left-Lower -> Right-Lower
  for (int idx = 1; idx<num_split; ++idx){
    angle = angle - sector_size/num_split;
    corner.x = r_len*cos(angle); corner.y = r_len*sin(angle);
    corner.z = calc_pt_z(corner.x, corner.y);
    polygons.polygon.points.push_back(corner);
  }

  return polygons;
}

void Patchwork_M::extract_initial_seeds_(const int zone_idx, const pcl::PointCloud<pcl::PointXYZI>& p_sorted,
                                       pcl::PointCloud<pcl::PointXYZI>& init_seeds){
  //function for non-uniform mode
  init_seeds.points.clear();

  // LPR is the mean of Low Point Representative
  double sum = 0;
  int cnt = 0;

  int init_idx = 0;
  if (zone_idx == 0){
    for(int i=0;i<p_sorted.points.size();i++){
      // do not set the initial seeds by the points under expected ground.
      if(p_sorted.points[i].z < adaptive_seed_selection_margin_ * sensor_height_){
        ++init_idx;
      } else{
        break;
      }
    }
  }

  // Calculate the mean height value.
  for(int i=init_idx;i<p_sorted.points.size() && cnt< num_lprs_set_[zone_idx] ;i++){
    sum += p_sorted.points[i].z;
    cnt++;
  }
  double lpr_height = cnt!=0?sum/cnt:0;// in case divide by 0

  // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
  for(int i=0;i<p_sorted.points.size();i++){
    if(p_sorted.points[i].z < lpr_height + th_seeds_){
      init_seeds.points.push_back(p_sorted.points[i]);
    }
  }
}


void Patchwork_M::extract_patchwiseground(const int zone_idx, const pcl::PointCloud<pcl::PointXYZI>& src,
                                        pcl::PointCloud<pcl::PointXYZI>& dst,
                                        pcl::PointCloud<pcl::PointXYZI>& non_ground_dst){
  //function for non-uniform mode
  // 0. Initialization
  if (!ground_pc_.empty()) ground_pc_.clear();
  if (!dst.empty()) dst.clear();
  if (!non_ground_dst.empty()) non_ground_dst.clear();
  

  // 1. set seeds!
  extract_initial_seeds_(zone_idx, src, ground_pc_);

  // 2. Extract ground
  for(int i=0; i<num_iter_; i++){
    estimate_plane_(ground_pc_);
    ground_pc_.clear();

    // pointcloud to matrix (Nx3 Matrix)
    Eigen::MatrixXf points(src.points.size(),3);
    int j = 0;
    for (auto& p:src.points){
      points.row(j++)<<p.x,p.y,p.z;
    }

    // ground plane model (Nx3)*(3x1) = Nx1
    Eigen::VectorXf result = points*normal_;

    // threshold filter
    for (int r=0; r<result.rows(); r++){
      if (i<num_iter_-1){
        if(result[r]<th_dist_d_){
          ground_pc_.points.push_back(src[r]);
        }
      } else{
        // Final iteration
        if(result[r]<th_dist_d_){
          dst.points.push_back(src[r]);
        } else{
          if (i==num_iter_-1){
              non_ground_dst.push_back(src[r]);
          }
        }
      }
    }
  }
}

void Patchwork_M::extract_patchwiseground(const int zone_idx, const pcl::PointCloud<pcl::PointXYZI>& src,
                                        pcl::PointCloud<pcl::PointXYZI>& dst,
                                        pcl::PointCloud<pcl::PointXYZI>& non_ground_dst,
                                        pcl::PointCloud<pcl::PointXYZI>& obstacle_dst){
  //function for non-uniform mode
  // 0. Initialization
  if (!ground_pc_.empty()) ground_pc_.clear();
  if (!dst.empty()) dst.clear();
  if (!non_ground_dst.empty()) non_ground_dst.clear();
  if (!obstacle_dst.empty()) obstacle_dst.clear();

  // 1. set seeds!
  extract_initial_seeds_(zone_idx, src, ground_pc_);

  // 2. Extract ground
  for(int i=0; i<num_iter_; i++){
    estimate_plane_(ground_pc_);
    ground_pc_.clear();

    // pointcloud to matrix (Nx3 Matrix)
    Eigen::MatrixXf points(src.points.size(),3);
    int j = 0;
    for (auto& p:src.points){
      points.row(j++)<<p.x,p.y,p.z;
    }

    // ground plane model (Nx3)*(3x1) = Nx1
    Eigen::VectorXf result = points*normal_;

    // threshold filter
    for (int r=0; r<result.rows(); r++){
      if (i<num_iter_-1){
        if(result[r]<th_dist_d_){
          ground_pc_.points.push_back(src[r]);
        }
      } else{
        // Final iteration
        if(result[r]<th_dist_d_){
          dst.points.push_back(src[r]);
        } else{
          if (i==num_iter_-1){
              if (result[r] < th_dist_obstacle_d_)
                obstacle_dst.push_back(src[r]);
              non_ground_dst.push_back(src[r]);
          }
        }
      }
    }
  }
}

geometry_msgs::PolygonStamped Patchwork_M::set_polygons(int zone_idx, int r_idx, int theta_idx, int num_split){
  //function for non-uniform mode
  //num_split is defined to make curved polygon.
  geometry_msgs::PolygonStamped polygons;

  // Set point of polygon. Start from Right-Lower and CCW.
  geometry_msgs::Point32 corner;
  double zone_min_range = min_ranges[zone_idx];

  //Right-Lower
  double r_len = r_idx * ring_sizes[zone_idx] + zone_min_range;
  double angle = theta_idx * sector_sizes[zone_idx];

  corner.x = r_len*cos(angle); corner.y = r_len*sin(angle);
  corner.z = calc_pt_z(corner.x, corner.y);
  polygons.polygon.points.push_back(corner);

  //Right-Upper
  r_len = r_len + ring_sizes[zone_idx];

  corner.x = r_len*cos(angle); corner.y = r_len*sin(angle);
  corner.z = calc_pt_z(corner.x, corner.y);
  polygons.polygon.points.push_back(corner);

  //Right-Upper -> Left-Upper
  for (int idx = 1; idx <= num_split; ++idx){
    angle = angle + sector_sizes[zone_idx]/num_split;
    corner.x = r_len*cos(angle); corner.y = r_len*sin(angle);
    corner.z = calc_pt_z(corner.x, corner.y);
    polygons.polygon.points.push_back(corner);
  }

  //Left-Lower
  r_len = r_len - ring_sizes[zone_idx];

  corner.x = r_len*cos(angle); corner.y = r_len*sin(angle);
  corner.z = calc_pt_z(corner.x, corner.y);
  polygons.polygon.points.push_back(corner);

  //Left-Lower -> Right_Lower
  for (int idx = 1; idx <num_split; ++idx){
    angle = angle - sector_sizes[zone_idx]/num_split;
    corner.x = r_len*cos(angle); corner.y = r_len*sin(angle);
    corner.z = calc_pt_z(corner.x, corner.y);
    polygons.polygon.points.push_back(corner);
  }


  return polygons;
}


#endif
