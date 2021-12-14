#include <ros/ros.h>
#include <derived_object_msgs/ObjectArray.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <ros/console.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_msgs/TFMessage.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <thread>
#include <deque>
#include <mutex>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <experimental/filesystem>
#include <signal.h>
#include <map>
#include <set>
#include <numeric>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
// #include <pcl/kdtree/kdtree_flann.h>
// #ifdef PCL_NO_PRECOMPILE
#include <pcl/kdtree/impl/kdtree_flann.hpp>
// #endif
#include "iou.h"

using namespace std;

struct CarlaPointXYZCIT {
    PCL_ADD_POINT4D;
    float CosAngle;
    uint32_t ObjIdx;
    uint32_t ObjTag;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(CarlaPointXYZCIT,
    (float, x, x) (float, y, y) (float, z, z) (float, CosAngle, CosAngle)
    (uint32_t, ObjIdx, ObjIdx) (uint32_t, ObjTag, ObjTag) 
)

struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

using PointXYZIRT = VelodynePointXYZIRT;
using PointCarla = CarlaPointXYZCIT;

class Track 
{
    public:
        bool seen;
        uint32_t gt_label;
        int pred_label;
        int type;
        int out_cnt;
        int in_cnt;
        int trk_cnt;
        int max_trk_cnt;
        float trk_ratio;
        
        Track(uint32_t _gt_label, int _type): pred_label(-1), out_cnt(0), in_cnt(0), trk_cnt(0), max_trk_cnt(0), trk_ratio(0), seen(false)
        {
            gt_label = _gt_label;
            type = _type;
        }

        Track() {}

        ~Track() {}

        void print() {
            cout << "Label: " << gt_label << " Type: " << type << endl;
            cout << "Pred: " << pred_label << " in_cnt: " << in_cnt << " out_cnt: " << out_cnt << " trk_cnt: " << trk_cnt << " ratio: " << trk_ratio << endl;
        }

        void reset() { // keep trk_ratio for evaluation
            trk_cnt = 0;
            in_cnt = 0;
            max_trk_cnt = 0;
            pred_label = -1;
        }
};

class Statistics
{
    public: 

        // Tracking
        std::vector<int> FN;
        std::vector<int> FP;
        std::vector<int> TP;
        std::vector<int> SWC;
        std::vector<int> GT;
        std::vector<float> MOTA;
        std::vector<int> FRAG;
        std::vector<float> trk_ratio;
        int t_FN = 0, t_FP = 0, t_SWC = 0, t_GT = 0, t_TP = 0;
        float t_MOTA = 0.0;
        int t_FRAG = 0;
        int MT = 0; // mostly tracked
        int PT = 0; // partially tracked
        int ML = 0; // mostly lost
        
        // Clustering
        std::vector<float> GTR_entropies;
        std::vector<float> PR_entropies;
    
        Statistics() {}


};

class Experiment
{
    public:

        tf::Transform t_ego_lidar;
        tf::StampedTransform t_world_start;
        
        // params
        string slam_save_dir;
        string trk_save_dir;
        string cls_save_dir;
        string dyn_save_dir;
        string cluster_method;
        string odom_topic;
        string point_topic;
        bool eval_slam;
        bool eval_tracking;
        bool eval_dynamic;
        bool eval_clustering;

        // string SAVE_DIR = "/home/euigon/experiment/traj_result/";
        string gt_file = "stamped_groundtruth.txt";
        string est_file = "stamped_traj_estimate.txt";
        vector<nav_msgs::Odometry> estimated_poses;
        vector<nav_msgs::Odometry> groundtruth_poses;
        bool first_odom_in;

        // Tracking
        ros::Publisher pub_trk_est_bbox;
        ros::Subscriber sub_est_bbox;
        ros::Subscriber sub_gt_bbox;
        ros::Publisher pub_gt_bbox;
        ros::Publisher pub_trk_cloud;
        ros::Publisher pub_cmp_bbox;
        ros::Subscriber sub_labeled_cloud;

        // SLAM
        ros::Publisher pub_gt_odom;
        ros::Publisher pub_estimated_odom;
        ros::Subscriber sub_estimated_odom;
        
        // semantic
        ros::Subscriber sub_tf;

        // Raw
        ros::Subscriber sub_raw_cloud;

        // queues
        deque<geometry_msgs::TransformStamped> gt_odom_queue;
        deque<nav_msgs::Odometry> est_odom_queue;
        deque<jsk_recognition_msgs::BoundingBoxArray> est_bbox_queue;
        deque<jsk_recognition_msgs::BoundingBoxArray> gt_bbox_queue;
        deque<sensor_msgs::PointCloud2> cloud_queue;
        
        tf::Transform t_map_odom;
        map<uint32_t, Track> track_table;

        // dyanmic classification
        ros::Subscriber sub_kitti_gt_bbox;
        string data_type;

        // tracking params
        float lidar_scope;
        int max_out_cnt;
        int min_pt_cnt;
        float gnd_ratio;

        Statistics stat;

        // clustering
        pcl::KdTreeFLANN<PointCarla>::Ptr kdtree;

        map<uint8_t, uint8_t> object_cnt;
        int frame_cnt;


    Experiment() {

        ros::NodeHandle nh("~");

        first_odom_in = false;

        nh.param<string>("experiment/slam_save_dir", slam_save_dir, "/home/euigon/experiment/traj_result/");
        nh.param<string>("experiment/trk_save_dir", trk_save_dir, "/home/euigon/experiment/trk_result/");
        nh.param<string>("experiment/dyn_save_dir", dyn_save_dir, "/home/euigon/experiment/trk_result/");

        nh.param<string>("rosbag/type", data_type, "");
        nh.param<string>("lio_sam/pointCloudTopic", point_topic, "");
        nh.param<string>("experiment/cluster_save_dir", cls_save_dir, "");
        nh.param<string>("tracking/clustering/method", cluster_method, "");
        nh.param<string>("lio_sam/odomTopic", odom_topic, "odometry");
        nh.param<bool>("experiment/eval_tracking", eval_tracking, false);
        nh.param<bool>("experiment/eval_clustering", eval_clustering, false);
        nh.param<bool>("experiment/eval_slam", eval_slam, false);
        nh.param<bool>("experiment/eval_dynamic", eval_dynamic, false);

        nh.param<float>("experiment/lidar_scope", lidar_scope, 40.0);
        nh.param<int>("experiment/max_out_cnt", max_out_cnt, 10);
        nh.param<int>("experiment/min_pt_cnt", min_pt_cnt, 10);
        nh.param<float>("experiment/gnd_ratio", gnd_ratio, 0.7);

        t_ego_lidar = tf::Transform(tf::createQuaternionFromYaw(0.0), tf::Vector3(-0.2, 0.0, 1.9));        
        
        // // tracking
        pub_trk_est_bbox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/exp/trk/est_bbox", 1);
        pub_trk_cloud = nh.advertise<sensor_msgs::PointCloud2>("/exp/trk/cloud", 1);
        sub_est_bbox = nh.subscribe<jsk_recognition_msgs::BoundingBoxArray>("/exp/est_bbox", 2000, &Experiment::estBoundingBoxCallback, this, ros::TransportHints().tcpNoDelay());
        sub_gt_bbox = nh.subscribe<derived_object_msgs::ObjectArray>("/carla/objects", 2000, &Experiment::objectCallback, this, ros::TransportHints().tcpNoDelay());
        sub_kitti_gt_bbox = nh.subscribe<jsk_recognition_msgs::BoundingBoxArray>("/objects", 2000, &Experiment::kittiObjectCallback, this, ros::TransportHints().tcpNoDelay());
        pub_gt_bbox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/exp/gt_bbox", 1);
        pub_cmp_bbox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/exp/cmp_bbox", 1);

        // slam
        pub_gt_odom = nh.advertise<nav_msgs::Odometry>("/exp/gt_odom", 1);
        pub_estimated_odom = nh.advertise<nav_msgs::Odometry>("/exp/est_odom", 1);
        sub_estimated_odom = nh.subscribe<nav_msgs::Odometry>("/lio_sam/mapping/odometry", 2000, &Experiment::estimatedOdomCallback, this, ros::TransportHints().tcpNoDelay());
        
        // semantic
        sub_raw_cloud = nh.subscribe<sensor_msgs::PointCloud2>(point_topic, 2000, &Experiment::cloudCallback, this, ros::TransportHints().tcpNoDelay());
        sub_tf = nh.subscribe<tf2_msgs::TFMessage>("/tf", 2000, &Experiment::tfCallback, this, ros::TransportHints().tcpNoDelay());
        
        // clustering
        sub_labeled_cloud = nh.subscribe<sensor_msgs::PointCloud2>("/tracking/lidar/meas_cluster_local", 2000, &Experiment::labeledCloudCallback, this, ros::TransportHints().tcpNoDelay());

        if (eval_slam) {
            ofstream gt_output(slam_save_dir + gt_file);
            ofstream est_output(slam_save_dir + est_file);
            gt_output.close();
            est_output.close();
        }

        if (eval_tracking) {
            ofstream trk_output(trk_save_dir + "result.txt");
            trk_output.close();
        }

        if (eval_clustering) {
            ofstream cls_output(cls_save_dir + cluster_method +"_entropy.csv");
            cls_output.close();
        }

        if (eval_dynamic) {
            ofstream dyn_output(dyn_save_dir + "result.csv");
            dyn_output.close();
        }

        // allocate memory
        kdtree.reset(new pcl::KdTreeFLANN<PointCarla>());
        frame_cnt = 0;
    }

    ~Experiment() {
        if (eval_slam) {
            std::cout<<"\033[1;32m"<< "You pressed Ctrl+C..." <<"\033[0m"<<std::endl;
            std::cout<<"\033[1;32m"<< "Saving to " << slam_save_dir << "\033[0m"<<std::endl;

            std::ofstream est_output(slam_save_dir + est_file, std::ios::app);
            std::ofstream gt_output(slam_save_dir + gt_file, std::ios::app);

            for (nav_msgs::Odometry pose_it : groundtruth_poses){

                std_msgs::Header header = pose_it.header;
                geometry_msgs::Point tl = pose_it.pose.pose.position;
                geometry_msgs::Quaternion qt = pose_it.pose.pose.orientation;

                gt_output<<header.stamp<<" "<<tl.x<<" "<<tl.y<<" "<<tl.z<< " "<<qt.x <<" "<< qt.y<<" "<<qt.z<<" "<<qt.w;
                gt_output<<std::endl;
            }

            for (nav_msgs::Odometry pose_it : estimated_poses){

                std_msgs::Header header = pose_it.header;
                geometry_msgs::Point tl = pose_it.pose.pose.position;
                geometry_msgs::Quaternion qt = pose_it.pose.pose.orientation;

                est_output<<header.stamp<<" "<<tl.x<<" "<<tl.y<<" "<<tl.z<< " "<<qt.x <<" "<< qt.y<<" "<<qt.z<<" "<<qt.w;
                est_output<<std::endl;
            }

            gt_output.close();
            est_output.close();

            std::cout<<"\033[1;32m"<< "Done..." <<"\033[0m"<<std::endl;

            exit(1);
        }

        if (eval_tracking) {
            std::cout<<"\033[1;32m"<< "You pressed Ctrl+C..." <<"\033[0m"<<std::endl;
            std::cout<<"\033[1;32m"<< "Saving to " << trk_save_dir << "\033[0m"<<std::endl;

            std::ofstream trk_output(trk_save_dir + "result.txt", std::ios::app);
            
            for (size_t i = 0; i < stat.FN.size(); i++) {
                trk_output << stat.FN[i]<<" "<<stat.FP[i]<<" "<< stat.SWC[i]<<" "<<stat.GT[i]<<" "<<stat.MOTA[i]<<" "<<stat.FRAG[i];
                trk_output << std::endl;
            }
            trk_output << "CLEAR metric: " << std::endl;
            trk_output << stat.t_FN<<" "<<stat.t_FP<<" "<<stat.t_SWC<<" "<<stat.t_GT<<" "<< 1-(float)(stat.t_FP+stat.t_FN+stat.t_SWC)/stat.t_GT << std::endl;
            
            cout << "Frames: " << frame_cnt << endl;
            cout << "Seen: " << endl;
            for (auto track : track_table) {
                if (track.second.seen) {
                    std::cout << track.second.gt_label << std::endl;
                    if (track.second.trk_ratio >= 0.8)
                        stat.MT++;
                    else if (track.second.trk_ratio >= 0.2 && track.second.trk_ratio < 0.8) 
                        stat.PT++;
                    else if (track.second.trk_ratio < 0.2 && track.second.trk_ratio >= 0)
                        stat.ML++;
                }
            }
            trk_output << "Quality: " << std::endl;
            trk_output << stat.MT << " " << stat.PT << " " << stat.ML << " " << stat.t_FRAG << std::endl;


            std::cout << "CLEAR metric: " << std::endl;
            std::cout << stat.t_FN<<" "<<stat.t_FP<<" "<<stat.t_SWC<<" "<<stat.t_GT<<" "<< 1-(float)(stat.t_FP+stat.t_FN+stat.t_SWC)/stat.t_GT << std::endl;
            std::cout << "Quality: " << std::endl;
            std::cout << stat.MT << " " << stat.PT << " " << stat.ML << " " << stat.t_FRAG << std::endl;

            trk_output.close();
            std::cout<<"\033[1;32m"<< "Done..." <<"\033[0m"<<std::endl;

            exit(1);
        }

        if (eval_dynamic) {
            std::cout<<"\033[1;32m"<< "You pressed Ctrl+C..." <<"\033[0m"<<std::endl;
            std::cout<<"\033[1;32m"<< "Saving to " << dyn_save_dir << "\033[0m"<<std::endl;

            std::ofstream dyn_output(dyn_save_dir + "result.csv", std::ios::app);

            std::vector<double> precisions;
            std::vector<double> recalls;
            double precision;
            double recall;
            for (size_t i = 0; i < stat.TP.size(); i++) {
                precision = stat.TP[i] / (1.0*(stat.TP[i] + stat.FP[i]));
                recall = stat.TP[i] / (1.0*(stat.TP[i] + stat.FN[i]));
                dyn_output << stat.TP[i] <<"," << stat.FP[i] << "," << stat.FN[i] << ",";
                dyn_output << precision << ","<< recall << endl;
                precisions.push_back(precision);
                recalls.push_back(recall);
            }
            dyn_output << std::accumulate(precisions.begin(), precisions.end(), 0.0) / stat.TP.size() << ","<< 
            std::accumulate(recalls.begin(), recalls.end(), 0.0) / stat.TP.size() << endl;

            dyn_output.close();
            std::cout<<"\033[1;32m"<< "Done..." <<"\033[0m"<<std::endl;
            exit(1);
        }

        if (eval_clustering) {
            ofstream cls_output(cls_save_dir + cluster_method +"_entropy.csv");
            assert(stat.GTR_entropies.size() == stat.PR_entropies.size());
            float count = static_cast<float>(stat.GTR_entropies.size());
            for (size_t i = 0; i < stat.GTR_entropies.size(); i++) {
                cls_output << i << "," << stat.GTR_entropies[i] <<","<<stat.PR_entropies[i] << std::endl;
            }
            cls_output << "avg" << "," << std::accumulate(stat.GTR_entropies.begin(), stat.GTR_entropies.end(), 0.0) / count << "," <<
            std::accumulate(stat.PR_entropies.begin(), stat.PR_entropies.end(), 0.0) / count << std::endl;
            cls_output.close();        
        }

    }
    void labeledCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg_in) {
        if (!eval_clustering)
            return;

        if (gt_odom_queue.empty() || gt_bbox_queue.empty() || cloud_queue.empty()) 
            return;

        // look for the gt odom for clustering
        while (gt_odom_queue.front().header.stamp < msg_in->header.stamp){
            gt_odom_queue.pop_front();                
        }
        if (gt_odom_queue.empty())
            return;
        
        // look for the gt bbox for clustering
        while (gt_bbox_queue.front().header.stamp < msg_in->header.stamp){
            gt_bbox_queue.pop_front();                
        }
        if (gt_bbox_queue.empty())
            return;

        // look for the cloud for clustering
        while (cloud_queue.front().header.stamp < msg_in->header.stamp){
            cloud_queue.pop_front();                
        }
        if (cloud_queue.empty())
            return;


        sensor_msgs::PointCloud2 current_cloud = *msg_in;
        sensor_msgs::PointCloud2 gt_cloud = cloud_queue.front();
        geometry_msgs::TransformStamped gt_odom = gt_odom_queue.front();
        jsk_recognition_msgs::BoundingBoxArray gt_bbox = gt_bbox_queue.front();

        assert(gt_odom.header.stamp.toSec() == current_cloud.header.stamp.toSec());
        assert(gt_bbox.header.stamp.toSec() == current_cloud.header.stamp.toSec());
        assert(gt_cloud.header.stamp.toSec() == current_cloud.header.stamp.toSec());

        // transform gt_bbox to "old_lidar_link" reference
        gt_bbox.header.frame_id = "old_lidar_link";
        for (size_t i = 0; i < gt_bbox.boxes.size(); i++) {
            gt_bbox.boxes[i].header.frame_id = "old_lidar_link";
            tf::Transform t_map_bbox;
            tf::Transform t_map_lidar;
            tf::poseMsgToTF(gt_bbox.boxes[i].pose, t_map_bbox);
            tf::transformMsgToTF(gt_odom.transform, t_map_lidar);
            tf::Transform t_lidar_bbox = t_map_lidar.inverse() * t_map_bbox;
            gt_bbox.boxes[i].pose.position.x = t_lidar_bbox.getOrigin().x();
            gt_bbox.boxes[i].pose.position.y = t_lidar_bbox.getOrigin().y();
            gt_bbox.boxes[i].pose.position.z = t_lidar_bbox.getOrigin().z();
            gt_bbox.boxes[i].pose.orientation.x = t_lidar_bbox.getRotation().x();
            gt_bbox.boxes[i].pose.orientation.y = t_lidar_bbox.getRotation().y();
            gt_bbox.boxes[i].pose.orientation.z = t_lidar_bbox.getRotation().z();
            gt_bbox.boxes[i].pose.orientation.w = t_lidar_bbox.getRotation().w();
        }

        // send "carla_map" to "old_lidar_link" transform
        gt_odom.child_frame_id = "old_lidar_link";
        static tf::TransformBroadcaster tf_broadcaster;
        tf_broadcaster.sendTransform(gt_odom);
         
        // publish
        current_cloud.header.frame_id = "old_lidar_link";

        if (pub_trk_cloud.getNumSubscribers() != 0)
            pub_trk_cloud.publish(current_cloud);
        if (pub_gt_bbox.getNumSubscribers() != 0)
            pub_gt_bbox.publish(gt_bbox);


        /* ----------------------
         * START EVALUATION (GTR)
         * ---------------------- */
        pcl::PointCloud<pcl::PointXYZI>::Ptr inside_points(new pcl::PointCloud<pcl::PointXYZI>());
        map<int, int> it_to_cnt; 
        float GTR_entropy = 0.0;
        for (auto& bbox : gt_bbox.boxes) {
            // get number of points inside bounding box
            inside_points->points.clear();
            it_to_cnt.clear();
            insideBBOX(current_cloud, bbox, *inside_points);
            int sum = 0;
            if (inside_points->size() >= min_pt_cnt) {
                // count labels inside the box
                for (auto &pt : inside_points->points) {
                    int label = (int)pt.intensity;
                    if (it_to_cnt.find(label) == it_to_cnt.end()) {
                        it_to_cnt.insert(std::make_pair(label, 1));
                    } else {
                        it_to_cnt[label]++;
                    }
                }

                // calculate entropy
                float entropy = 0.0;
                for (auto &element : it_to_cnt) {
                    entropy -= 1.0 * element.second / inside_points->size() * log(1.0*element.second / inside_points->size());
                } 
                GTR_entropy += entropy;
            }
        }
        printf("GTR: %f\n", GTR_entropy);
        stat.GTR_entropies.push_back(GTR_entropy);

        /* ----------------------
        * START EVALUATION (PR)
        * ---------------------- */
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<PointCarla>::Ptr gt_cloud_in(new pcl::PointCloud<PointCarla>());
        pcl::moveFromROSMsg(current_cloud, *cloud_in);
        pcl::moveFromROSMsg(gt_cloud, *gt_cloud_in);
        map<int, std::vector<int>> id_to_labels;
        float PR_entropy = 0.0;
        // create kd tree
        kdtree->setInputCloud(gt_cloud_in);

        for (auto &pt : cloud_in->points) {
            PointCarla search_pt;
            search_pt.x = pt.x;
            search_pt.y = pt.y;
            search_pt.z = pt.z;
            int id = (int)pt.intensity;

            if (id_to_labels.find(id) == id_to_labels.end()) {
                id_to_labels.emplace(id, std::vector<int>(24, 0));
            } 

            std::vector<int> point_indices;
            std::vector<float> point_dists;
            if (kdtree->nearestKSearch(search_pt, 1, point_indices, point_dists) > 0) {
                if (point_dists[0] < 1.0) {
                    uint32_t label = gt_cloud_in->points[point_indices[0]].ObjTag;
                    id_to_labels[id][label]++;
                    // printf("xyz: %f;%f;%f, id: %d, label: %d, cnt: %d\n", pt.x, pt.y, pt.z, id, label, id_to_labels[id][label]);
                } else {
                    printf("Too far\n");
                }
            }
        }

        for (const auto &element : id_to_labels) {
            std::vector<int> labels = element.second;
            float entropy = 0.0;
            float total = std::accumulate(labels.begin(), labels.end(), 0) * 1.0;
            if (total > 0) {
                for (size_t j = 0; j < labels.size(); j++) {
                    if (labels[j] > 0) {
                        entropy -= labels[j] / total * log(labels[j] / total); 
                        // printf("id: %d, label: %d, entropy: %f, total: %f\n", element.first, j, entropy, total);           
                    }
                }
            }
            PR_entropy += entropy;
        }
        printf("PR: %f\n", PR_entropy);
        stat.PR_entropies.push_back(PR_entropy);
    }

    void kittiObjectCallback(const jsk_recognition_msgs::BoundingBoxArray::ConstPtr &msg_in) {
        gt_bbox_queue.push_back(*msg_in);
    }

    void objectCallback(const derived_object_msgs::ObjectArray::ConstPtr &msg_in) {
        jsk_recognition_msgs::BoundingBoxArray bbox_array;
        bbox_array.header.stamp = msg_in->header.stamp;
        bbox_array.header.frame_id = "carla_map";
        object_cnt.clear();
        for (size_t i = 0; i < msg_in->objects.size(); i++) {
            derived_object_msgs::Object obj = msg_in->objects[i];
            if (object_cnt.find(obj.classification) == object_cnt.end()) {
                object_cnt.insert(std::make_pair(obj.classification, 1));
            } else {
                object_cnt[obj.classification]++;
            }
            jsk_recognition_msgs::BoundingBox bbox;
            bbox.label = obj.id;
            bbox.value = (float) obj.classification * 1.0;
            bbox.header.stamp = msg_in->header.stamp;
            bbox.header.frame_id = "carla_map";
            // set dimension
            if (obj.shape.dimensions.size() == 3) { // bounding box
                bbox.dimensions.x = obj.shape.dimensions[0];
                bbox.dimensions.y = obj.shape.dimensions[1];
                bbox.dimensions.z = obj.shape.dimensions[2];
            } else {
                cout << "Not a bounding-box" << endl;
            }

            // set pose
            bbox.pose = obj.pose;
            if (obj.classification != derived_object_msgs::Object::CLASSIFICATION_PEDESTRIAN) {
                bbox.pose.position.z += obj.shape.dimensions[2] / 2.0;
            }
            bbox_array.boxes.push_back(bbox);
        }
        gt_bbox_queue.push_back(bbox_array);
        // Statistics for object numbers
        // printf("Object stat: \n");
        // for (auto element : object_cnt) {
        //     printf("ID: %d -- %d\n", element.first, element.second);
        // }
    }

    void tfCallback(const tf2_msgs::TFMessageConstPtr &msg_in) {
        tf2_msgs::TFMessage cur_tf = *msg_in;
        // 0.9.10
        // for (const auto ts_msg : msg_in->transforms) {
        //     if (ts_msg.header.frame_id == "map" && ts_msg.child_frame_id == "ego_vehicle/semantic_lidar") {
        //         cout << "TF received" << endl;
        //         gt_odom_queue.push_back(ts_msg);
        //     }
        // }

        // 0.9.12
        for (auto ts_msg : cur_tf.transforms) {
            if (ts_msg.header.frame_id == "map" && ts_msg.child_frame_id == "ego_vehicle") {
                tf::Transform t_map_ego;
                tf::transformMsgToTF(ts_msg.transform, t_map_ego);
                tf::Transform t_map_lidar = t_map_ego * t_ego_lidar;

                // find map to odom transformation
                if (!first_odom_in) {
                    t_map_odom = t_map_lidar;
                    t_map_odom.setRotation(tf::createQuaternionFromYaw(0.0));
                    first_odom_in = true;
                }

                // send odom to map (permanent)
                static tf::TransformBroadcaster tf_broadcaster;
                tf::StampedTransform stamped_odom_map = tf::StampedTransform(t_map_odom.inverse(), ts_msg.header.stamp, "odom", "carla_map");
                tf_broadcaster.sendTransform(stamped_odom_map);

                geometry_msgs::Transform final_trans;
                tf::transformTFToMsg(t_map_lidar, final_trans);
                ts_msg.transform = final_trans;
                ts_msg.header.frame_id = "carla_map";
                gt_odom_queue.push_back(ts_msg);
            }
        }
    }

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg_in) {
        // 2. convert current semantic point cloud to pcl
        pcl::PointCloud<PointCarla>::Ptr carla_cloud(new pcl::PointCloud<PointCarla>());
        sensor_msgs::PointCloud2 current_cloud = *msg_in;
        cloud_queue.push_back(current_cloud);
        // pcl::moveFromROSMsg(current_cloud, *carla_cloud);

        // cout << "Time: " << msg_in->header.stamp.toSec() << endl;
        // for (auto pt : carla_cloud->points) {
        //     // for (const auto bbox : cur_bbox_array.boxes) {
        //     // }
        //     if (pt.ObjIdx > 0) {
        //         // cout << pt.x << ";" << pt.y << ";" << pt.z << " idx: " << pt.ObjIdx << " tag: " << pt.ObjTag << endl;
        //     }
        // }
    }

    void estBoundingBoxCallback(const jsk_recognition_msgs::BoundingBoxArray::ConstPtr &msg_in) {
        if (eval_tracking || eval_dynamic) {
            jsk_recognition_msgs::BoundingBoxArray cur_bbox = *msg_in;
            est_bbox_queue.push_back(cur_bbox);
            if (data_type == "carla")
                tracking_evaluate_carla();
            else if (data_type == "kitti")
                tracking_evaluate_kitti();
        }
    }

    void estimatedOdomCallback(const nav_msgs::Odometry::ConstPtr &msg_in) {
        if (eval_slam) {
            nav_msgs::Odometry cur_odom = *msg_in;
            est_odom_queue.push_back(cur_odom);
            slam_evaluate();
        }
    }

   void tracking_evaluate_kitti() {
        printf("---------EVALUATING KITTI DATASET-------------\n");
        printf("est bbox: %d, cloud_queue: %d, gt bbox: %d\n", (int)est_bbox_queue.size(), (int)cloud_queue.size(), (int)gt_bbox_queue.size());

        if (est_bbox_queue.empty() || cloud_queue.empty() || gt_bbox_queue.empty()) 
            return;
        
        jsk_recognition_msgs::BoundingBoxArray cur_bbox_array = est_bbox_queue.front();
        est_bbox_queue.pop_front();

        // look for point cloud
        while (cloud_queue.front().header.stamp < cur_bbox_array.header.stamp) {
            cloud_queue.pop_front();
        }
        if (cloud_queue.empty())
            return;

        // look for the gt bbox for tracking
        while (gt_bbox_queue.front().header.stamp < cur_bbox_array.header.stamp){
            gt_bbox_queue.pop_front();                
        }
        if (gt_bbox_queue.empty())
            return;

        sensor_msgs::PointCloud2 current_cloud = cloud_queue.front();
        jsk_recognition_msgs::BoundingBoxArray gt_bbox = gt_bbox_queue.front();
        assert(current_cloud.front().header.stamp.toSec() == cur_bbox_array.header.stamp.toSec());
        assert(gt_bbox.header.stamp.toSec() == cur_bbox_array.header.stamp.toSec());
    
        // transform gt_bbox to "old_lidar_link" reference
        gt_bbox.header.frame_id = "old_lidar_link";
        for (size_t i = 0; i < gt_bbox.boxes.size(); i++) {
            gt_bbox.boxes[i].header.frame_id = "old_lidar_link";
        }

        // publish
        current_cloud.header.frame_id = "old_lidar_link";
        jsk_recognition_msgs::BoundingBoxArray est_dynamic_bbox;
        est_dynamic_bbox.header.frame_id = "old_lidar_link";
        for (size_t i = 0; i < cur_bbox_array.boxes.size(); i++) {
            cur_bbox_array.boxes[i].header.frame_id = "old_lidar_link";
            jsk_recognition_msgs::BoundingBox bbox = cur_bbox_array.boxes[i];
            if (bbox.value == 1.0) {
                // filter estimation not in camera view
                if (abs(bbox.pose.position.y / bbox.pose.position.x) > 0.86 || bbox.pose.position.x < 0 || abs(bbox.pose.position.z / bbox.pose.position.x) > 10)
                    continue;
                printf("est: %f;%f;%f\n", bbox.pose.position.x, bbox.pose.position.y, bbox.pose.position.z);
                est_dynamic_bbox.boxes.push_back(bbox);   
            }
        }

        if (pub_trk_cloud.getNumSubscribers() != 0)
            pub_trk_cloud.publish(current_cloud);
        if (pub_trk_est_bbox.getNumSubscribers() != 0)
            pub_trk_est_bbox.publish(est_dynamic_bbox);   
        if (pub_gt_bbox.getNumSubscribers() != 0)
            pub_gt_bbox.publish(gt_bbox);

        /* ----------------------
         * START EVALUATION
         * ---------------------- */
        int c_false_pos = 0;
        int c_true_pos = 0;
        int c_pos = 0;
        int c_false_neg = 0;

        jsk_recognition_msgs::BoundingBoxArray inside_scope_bbox;
        inside_scope_bbox.header = gt_bbox.header;

        map<uint32_t, bool> gt_checker;

        // Neagtives
        set<int> associated;
        for (auto gt : gt_bbox.boxes) {

            std::shared_ptr<Track> cur_tracker(new Track(gt.label, (int)gt.value));

            // get number of points inside bounding box
            pcl::PointCloud<PointXYZIRT>::Ptr inside_points(new pcl::PointCloud<PointXYZIRT>());
            
            insideBBOX(current_cloud, gt, *inside_points);
            // printf("Inside points num: %d\n", inside_points->size());

            if (inside_points->points.size() >= min_pt_cnt) { 
                inside_scope_bbox.boxes.push_back(gt);
                
                // check iou
                vector<jsk_recognition_msgs::BoundingBox> candidates;
                vector<double> ious;
                c_pos = 0;
                for (auto est: est_dynamic_bbox.boxes) {
                    if (est.value == 1.0) { // dynamic
                        c_pos++;
                        IOU::Vertexes gt_vertices;
                        IOU::Vertexes est_vertices;
                        bboxToVerticies(gt, gt_vertices);
                        bboxToVerticies(est, est_vertices);
                        double iou = IOU::iouEx(gt_vertices, est_vertices);
                        if (iou > 0) {
                            candidates.push_back(est);
                            ious.push_back(iou);
                            // cout << "Negatives" << endl;
                            // cout << "gt: " << gt.pose.position.x << ";"<<gt.pose.position.y << ";"<<gt.pose.position.z << endl;
                            // cout << "est: " << est.pose.position.x << ";"<<est.pose.position.y << ";"<<est.pose.position.z << endl;
                            // cout << "iou: " << iou << endl;
                        } 
                    }
                }
                if (candidates.size() == 0) {
                    c_false_neg++;
                } else if (candidates.size() > 0) {
                    c_true_pos++;
                }
            } 
        }
        // no predictions at all 
        if (c_pos == 0) {
            ROS_ERROR("No prediction");
            return;
        }
        

        c_false_pos = c_pos - c_true_pos;

        
        // gt_checker.clear();
        // // Positives
        // for (auto est : est_dynamic_bbox.boxes) {
        //     if (est.value == 1.0) { // dynamic
        //         // filter estimation not in camera view
        //         if (abs(est.pose.position.y / est.pose.position.x) > 0.86 || est.pose.position.x < 0 || abs(est.pose.position.z / est.pose.position.x) > 10)
        //             continue;

        //         c_pos++;

        //         for (auto gt: gt_bbox.boxes) {
        //             if (gt_checker.find(gt.label) != gt_checker.end())
        //                 continue;
        //             // get number of points inside bounding box
        //             pcl::PointCloud<PointXYZIRT>::Ptr inside_points(new pcl::PointCloud<PointXYZIRT>());
        //             insideBBOX(current_cloud, gt, *inside_points);
        //             // printf("Inside points num: %d\n", inside_points->size());

        //             if (inside_points->points.size() >= min_pt_cnt) { 

        //                 IOU::Vertexes gt_vertices;
        //                 IOU::Vertexes est_vertices;
        //                 bboxToVerticies(gt, gt_vertices);
        //                 bboxToVerticies(est, est_vertices);
        //                 double iou = IOU::iouEx(gt_vertices, est_vertices);
        //                 if (iou > 0) {
        //                     cout << "Positives" << endl;
        //                     cout << "gt: " << gt.pose.position.x << ";"<<gt.pose.position.y << ";"<<gt.pose.position.z << endl;
        //                     cout << "est: " << est.pose.position.x << ";"<<est.pose.position.y << ";"<<est.pose.position.z << endl;
        //                     cout << "iou: " << iou << endl;
        //                     c_true_pos++;
        //                     gt_checker.insert(std::make_pair(gt.label, true));
        //                 }
        //             }
        //         }
        //     }         
        // }
        // update overall metric
        stat.t_FN += c_false_neg;
        stat.t_FP += c_false_pos;
        stat.t_TP += c_true_pos;
        stat.TP.push_back(c_true_pos);
        stat.FN.push_back(c_false_neg);
        stat.FP.push_back(c_false_pos);
        cout << "FN: " << c_false_neg << " FP: " << c_false_pos << " TP: " << c_true_pos << " Total pos:" << c_pos << endl;
        cout << c_true_pos / (1.0*(c_pos)) << ","<< c_true_pos / (1.0*(c_true_pos + c_false_neg)) << endl;
        if (pub_cmp_bbox.getNumSubscribers() != 0)
            pub_cmp_bbox.publish(inside_scope_bbox);
    }

    void tracking_evaluate_carla() {

        /* -------------------------
         * RETRIEVE DATA & VISUALIZE
         * ------------------------- */

        if (est_bbox_queue.empty() || cloud_queue.empty() || gt_odom_queue.empty() || gt_bbox_queue.empty()) 
            return;
        
        jsk_recognition_msgs::BoundingBoxArray cur_bbox_array = est_bbox_queue.front();
        est_bbox_queue.pop_front();

        // look for point cloud
        while (cloud_queue.front().header.stamp < cur_bbox_array.header.stamp) {
            cloud_queue.pop_front();
        }
        if (cloud_queue.empty())
            return;

        // look for the gt odom for tracking
        while (gt_odom_queue.front().header.stamp < cur_bbox_array.header.stamp){
            gt_odom_queue.pop_front();                
        }
        if (gt_odom_queue.empty())
            return;
        
        // look for the gt bbox for tracking
        while (gt_bbox_queue.front().header.stamp < cur_bbox_array.header.stamp){
            gt_bbox_queue.pop_front();                
        }
        if (gt_bbox_queue.empty())
            return;
        
        sensor_msgs::PointCloud2 current_cloud = cloud_queue.front();
        geometry_msgs::TransformStamped gt_odom = gt_odom_queue.front();
        jsk_recognition_msgs::BoundingBoxArray gt_bbox = gt_bbox_queue.front();

        assert(current_cloud.front().header.stamp.toSec() == cur_bbox_array.header.stamp.toSec());
        assert(gt_odom.header.stamp.toSec() == cur_bbox_array.header.stamp.toSec());
        assert(gt_bbox.header.stamp.toSec() == cur_bbox_array.header.stamp.toSec());

        // transform gt_bbox to "old_lidar_link" reference
        gt_bbox.header.frame_id = "old_lidar_link";
        for (size_t i = 0; i < gt_bbox.boxes.size(); i++) {
            gt_bbox.boxes[i].header.frame_id = "old_lidar_link";
            tf::Transform t_map_bbox;
            tf::Transform t_map_lidar;
            tf::poseMsgToTF(gt_bbox.boxes[i].pose, t_map_bbox);
            tf::transformMsgToTF(gt_odom.transform, t_map_lidar);
            tf::Transform t_lidar_bbox = t_map_lidar.inverse() * t_map_bbox;
            gt_bbox.boxes[i].pose.position.x = t_lidar_bbox.getOrigin().x();
            gt_bbox.boxes[i].pose.position.y = t_lidar_bbox.getOrigin().y();
            gt_bbox.boxes[i].pose.position.z = t_lidar_bbox.getOrigin().z();
            gt_bbox.boxes[i].pose.orientation.x = t_lidar_bbox.getRotation().x();
            gt_bbox.boxes[i].pose.orientation.y = t_lidar_bbox.getRotation().y();
            gt_bbox.boxes[i].pose.orientation.z = t_lidar_bbox.getRotation().z();
            gt_bbox.boxes[i].pose.orientation.w = t_lidar_bbox.getRotation().w();
        }

        // send "carla_map" to "old_lidar_link" transform
        gt_odom.child_frame_id = "old_lidar_link";
        static tf::TransformBroadcaster tf_broadcaster;
        tf_broadcaster.sendTransform(gt_odom);
         
        // publish
        current_cloud.header.frame_id = "old_lidar_link";
        cur_bbox_array.header.frame_id = "old_lidar_link";
        for (size_t i = 0; i < cur_bbox_array.boxes.size(); i++) {
            cur_bbox_array.boxes[i].header.frame_id = "old_lidar_link";
        }

        if (pub_trk_cloud.getNumSubscribers() != 0)
            pub_trk_cloud.publish(current_cloud);
        if (pub_trk_est_bbox.getNumSubscribers() != 0)
            pub_trk_est_bbox.publish(cur_bbox_array);   
        if (pub_gt_bbox.getNumSubscribers() != 0)
            pub_gt_bbox.publish(gt_bbox);

        /* ----------------------
         * START EVALUATION
         * ---------------------- */
        int c_fragmentation = 0;
        int c_false_pos = 0;
        int c_false_neg = 0;
        int c_id_switch = 0; 
        int c_gt_cnt = 0;

        jsk_recognition_msgs::BoundingBoxArray inside_scope_bbox;
        inside_scope_bbox.header = gt_bbox.header;

        set<int> associated;

        for (auto gt : gt_bbox.boxes) {
            float dist = sqrt(
                        (gt.pose.position.x) * (gt.pose.position.x) +
                        (gt.pose.position.y) * (gt.pose.position.y) +
                        (gt.pose.position.z) * (gt.pose.position.z)); 
                        
            std::shared_ptr<Track> cur_tracker(new Track(gt.label, (int)gt.value));

            auto it = track_table.find(gt.label);
            if (it != track_table.end()) // exist
                *cur_tracker = track_table[gt.label];
            
            // get number of points inside bounding box
            pcl::PointCloud<PointCarla>::Ptr inside_points(new pcl::PointCloud<PointCarla>());
            insideBBOX(current_cloud, gt, *inside_points);

            if (dist <= lidar_scope && dist >= 1.5 && inside_points->points.size() >= min_pt_cnt) { 
                inside_scope_bbox.boxes.push_back(gt);
                c_gt_cnt++;
                cur_tracker->seen = true;
                cur_tracker->in_cnt++;
                cur_tracker->out_cnt = 0;
                
                // check iou
                vector<jsk_recognition_msgs::BoundingBox> candidates;
                vector<double> ious;
                for (auto est: cur_bbox_array.boxes) {
                    IOU::Vertexes gt_vertices;
                    IOU::Vertexes est_vertices;
                    bboxToVerticies(gt, gt_vertices);
                    bboxToVerticies(est, est_vertices);
                    double iou = IOU::iouEx(gt_vertices, est_vertices);
                    if (iou > 0) {
                        candidates.push_back(est);
                        ious.push_back(iou);
                        // cout << "gt: " << gt.pose.position.x << ";"<<gt.pose.position.y << ";"<<gt.pose.position.z << endl;
                        // cout << "est: " << est.pose.position.x << ";"<<est.pose.position.y << ";"<<est.pose.position.z << endl;
                        // cout << "iou: " << iou << endl;
                        associated.insert(est.label);
                    }
                }

                // one or more candidates inside gt_bbox
                // cout << "gt: " << gt.pose.position.x << ";"<<gt.pose.position.y << ";"<<gt.pose.position.z << endl;
                if (candidates.size() >= 1) {
                    if (candidates.size() > 1) {
                        c_fragmentation += (candidates.size() - 1);
                        c_false_pos += (candidates.size() - 1); // include this?
                    }
                    
                    // set id of highest iou bbox to gt_bbox's id
                    int max_idx = std::max_element(ious.begin(), ious.end()) - ious.begin();
                    
                    if (cur_tracker->pred_label != candidates[max_idx].label) { // id switch
                        c_id_switch++;
                        if (cur_tracker->trk_cnt > cur_tracker->max_trk_cnt) { // update max tracking cnt (correct)
                            cur_tracker->max_trk_cnt = cur_tracker->trk_cnt;
                        }
                        cur_tracker->max_trk_cnt += cur_tracker->trk_cnt; // wrong
                        cur_tracker->trk_cnt = 0;
                    } else {
                        cur_tracker->trk_cnt++;
                        if (cur_tracker->trk_cnt > cur_tracker->max_trk_cnt) { // update max tracking cnt (correct)
                            cur_tracker->max_trk_cnt = cur_tracker->trk_cnt;
                        }
                        cur_tracker->max_trk_cnt += cur_tracker->trk_cnt; // wrong
                    }
                    cur_tracker->pred_label = candidates[max_idx].label;
                } else { // no est_bbox associated
                    c_false_neg++;
                }
                cur_tracker->trk_ratio = cur_tracker->max_trk_cnt / (float)cur_tracker->in_cnt;
                // cur_tracker->print();
            } else { // outside tracking range
                cur_tracker->out_cnt++;
                if (cur_tracker->out_cnt > max_out_cnt) { // ~2 seconds -> reset
                    cur_tracker->reset();
                }                    
            }
            track_table[gt.label] = *cur_tracker;
        }
        
        // False positive (having bbox on majority of ground points)
        for (auto est : cur_bbox_array.boxes) {
            auto it = associated.find(est.label);
            int gnd_cnt = 0; // # of ground lidar points
            if (it == associated.end()) { // not associated
                pcl::PointCloud<PointCarla>::Ptr points_inside(new pcl::PointCloud<PointCarla>());
                insideBBOX(current_cloud, est, *points_inside);
                for (const auto pt : points_inside->points) {
                    if (pt.ObjTag == 6u || pt.ObjTag == 7u || pt.ObjTag == 8u || pt.ObjTag == 14u) 
                        gnd_cnt++;
                }
                float ratio = (float) gnd_cnt / points_inside->points.size();
                if (ratio >= gnd_ratio)
                    c_false_pos++;
            }
        }

        // update overall metric
        stat.t_FN += c_false_neg;
        stat.t_FP += c_false_pos;
        stat.t_FRAG += c_fragmentation;
        stat.t_SWC += c_id_switch;
        stat.t_GT += c_gt_cnt;
        frame_cnt++;
        if (c_gt_cnt > 0) {
            stat.FN.push_back(c_false_neg);
            stat.FP.push_back(c_false_pos);
            stat.FRAG.push_back(c_fragmentation);
            stat.SWC.push_back(c_id_switch);
            stat.GT.push_back(c_gt_cnt);
            stat.MOTA.push_back(1 - float(c_false_pos + c_false_neg + c_id_switch) / c_gt_cnt);
            cout << "FN: " << c_false_neg << " FP: " << c_false_pos << " Frag: " << c_fragmentation << " Switch: " << c_id_switch << " gt: " << c_gt_cnt << endl;
            cout << "MOTA: " << 1-(c_false_pos + c_false_neg + c_id_switch) / (float)c_gt_cnt << endl;
        }
        if (pub_cmp_bbox.getNumSubscribers() != 0)
            pub_cmp_bbox.publish(inside_scope_bbox);
    }

    template <typename PointT>
    void insideBBOX(sensor_msgs::PointCloud2 this_cloud, jsk_recognition_msgs::BoundingBox this_bbox, typename pcl::PointCloud<PointT> &inside_points) {
        // 2. convert current semantic point cloud to pcl
        typename pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud<PointT>());
        pcl::moveFromROSMsg(this_cloud, *pcl_cloud);
        tf::Quaternion quat;
        tf::quaternionMsgToTF(this_bbox.pose.orientation, quat);        
        tf::Matrix3x3 rot_mat(quat);
        double roll, pitch, yaw;
        rot_mat.getRPY(roll, pitch, yaw);   
        
        float offset_x;
        float offset_y;
        float rot_x;
        float rot_y;
        int cnt = 0;
        for (auto pt : pcl_cloud->points) {
            offset_x = pt.x - this_bbox.pose.position.x;
            offset_y = pt.y - this_bbox.pose.position.y;
            rot_x = offset_x*cos(-yaw) - offset_y*sin(-yaw);
            rot_y = offset_x*sin(-yaw) + offset_y *cos(-yaw);
            if (rot_x >= -this_bbox.dimensions.x / 2.0 && 
                rot_x <= this_bbox.dimensions.x / 2.0 && 
                rot_y >= -this_bbox.dimensions.y / 2.0 &&
                rot_y <= this_bbox.dimensions.y / 2.0) {
                inside_points.points.push_back(pt);
            }
        }
        // cout << "Inside: " << endl;
        // cout << "Center: " << this_bbox.pose.position.x << " " << this_bbox.pose.position.y << " " << this_bbox.pose.position.z << " count: " << cnt << endl;
    }

    void bboxToVerticies(jsk_recognition_msgs::BoundingBox bbox, vector<IOU::Point> &vertices) {
        tf::Quaternion quat;
        tf::quaternionMsgToTF(bbox.pose.orientation, quat);
        tf::Matrix3x3 m(quat);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        // top right
        IOU::Point top_right;
        IOU::Point top_left;
        IOU::Point bottom_left;
        IOU::Point bottom_right;

        top_right.x = bbox.pose.position.x + (bbox.dimensions.x/2.0)*cos(yaw) - (bbox.dimensions.y/2.0)*sin(yaw);
        top_right.y = bbox.pose.position.y + (bbox.dimensions.x/2.0)*sin(yaw) + (bbox.dimensions.y/2.0)*cos(yaw);

        top_left.x = bbox.pose.position.x - (bbox.dimensions.x/2.0)*cos(yaw) - (bbox.dimensions.y/2.0)*sin(yaw);
        top_left.y = bbox.pose.position.y - (bbox.dimensions.x/2.0)*sin(yaw) + (bbox.dimensions.y/2.0)*cos(yaw);

        bottom_left.x = bbox.pose.position.x - (bbox.dimensions.x/2.0)*cos(yaw) + (bbox.dimensions.y/2.0)*sin(yaw);
        bottom_left.y = bbox.pose.position.y - (bbox.dimensions.x/2.0)*sin(yaw) - (bbox.dimensions.y/2.0)*cos(yaw);

        bottom_right.x = bbox.pose.position.x + (bbox.dimensions.x/2.0)*cos(yaw) + (bbox.dimensions.y/2.0)*sin(yaw);
        bottom_right.y = bbox.pose.position.y + (bbox.dimensions.x/2.0)*sin(yaw) - (bbox.dimensions.y/2.0)*cos(yaw);

        vertices.push_back(top_left);
        vertices.push_back(top_right);
        vertices.push_back(bottom_right);
        vertices.push_back(bottom_left);
    }  

    void slam_evaluate() {

        if (est_odom_queue.empty() || gt_odom_queue.empty()) 
            return;
        
        // Estimated pose
        nav_msgs::Odometry cur_odom_lidar;
        cur_odom_lidar = est_odom_queue.front();
        est_odom_queue.pop_front();
        
        // Find the gt pose for the first estimated pose
        while (gt_odom_queue.front().header.stamp < cur_odom_lidar.header.stamp){
            gt_odom_queue.pop_front();                
        }

        if (gt_odom_queue.empty())
            return;

        geometry_msgs::TransformStamped gt_odom = gt_odom_queue.front();
        assert(gt_odom.header.stamp.toSec() == cur_odom_lidar.header.stamp.toSec());

        tf::Transform t_map_lidar;
        tf::transformMsgToTF(gt_odom.transform, t_map_lidar);

        // push to estimated poses
        tf::Transform t_cur_odom_lidar;
        tf::poseMsgToTF(cur_odom_lidar.pose.pose, t_cur_odom_lidar);
        tf::Transform t_cur_map_lidar = t_map_odom * t_cur_odom_lidar;

        nav_msgs::Odometry cur_map_lidar;
        cur_map_lidar.header.stamp = cur_odom_lidar.header.stamp;
        cur_map_lidar.header.frame_id = "carla_map";
        cur_map_lidar.pose.pose.position.x = t_cur_map_lidar.getOrigin().x();
        cur_map_lidar.pose.pose.position.y = t_cur_map_lidar.getOrigin().y();
        cur_map_lidar.pose.pose.position.z = t_cur_map_lidar.getOrigin().z();
        cur_map_lidar.pose.pose.orientation.x = t_cur_map_lidar.getRotation().x();
        cur_map_lidar.pose.pose.orientation.y = t_cur_map_lidar.getRotation().y();
        cur_map_lidar.pose.pose.orientation.z = t_cur_map_lidar.getRotation().z();
        cur_map_lidar.pose.pose.orientation.w = t_cur_map_lidar.getRotation().w();
        estimated_poses.push_back(cur_map_lidar);
        pub_estimated_odom.publish(cur_map_lidar);

        // push to ground-truth poses
        nav_msgs::Odometry gt_map_lidar;
        gt_map_lidar.header.stamp = gt_odom.header.stamp;
        gt_map_lidar.header.frame_id = "carla_map";
        gt_map_lidar.pose.pose.position.x = t_map_lidar.getOrigin().x();
        gt_map_lidar.pose.pose.position.y = t_map_lidar.getOrigin().y();
        gt_map_lidar.pose.pose.position.z = t_map_lidar.getOrigin().z();
        gt_map_lidar.pose.pose.orientation.x = t_map_lidar.getRotation().x();
        gt_map_lidar.pose.pose.orientation.y = t_map_lidar.getRotation().y();
        gt_map_lidar.pose.pose.orientation.z = t_map_lidar.getRotation().z();
        gt_map_lidar.pose.pose.orientation.w = t_map_lidar.getRotation().w();
        groundtruth_poses.push_back(gt_map_lidar);
        pub_gt_odom.publish(gt_map_lidar); 
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "experiment");

    if( ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Error) ) {
        ros::console::notifyLoggerLevelsChanged();
    }

    ROS_INFO("\033[1;32m----> experiment started.\033[0m");

    Experiment exp;

    ros::MultiThreadedSpinner spinner(4);

    spinner.spin();

    return 0;
}
