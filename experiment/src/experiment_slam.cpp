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

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

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
        std::vector<int> FN;
        std::vector<int> FP;
        std::vector<int> SWC;
        std::vector<int> GT;
        std::vector<float> MOTA;
        std::vector<int> FRAG;
        std::vector<float> trk_ratio;
        int t_FN = 0, t_FP = 0, t_SWC = 0, t_GT = 0;
        float t_MOTA = 0.0;
        int t_FRAG = 0;
        int MT = 0; // mostly tracked
        int PT = 0; // partially tracked
        int ML = 0; // mostly lost
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
        string odom_topic;
        bool eval_slam;
        bool eval_tracking;

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

        // tracking params
        float lidar_scope;
        int max_out_cnt;
        int min_pt_cnt;

        Statistics stat;

    Experiment() {

        ros::NodeHandle nh("~");

        first_odom_in = false;

        nh.param<string>("experiment/slam_save_dir", slam_save_dir, "/home/euigon/experiment/traj_result/");
        nh.param<string>("experiment/trk_save_dir", trk_save_dir, "/home/euigon/experiment/trk_result/");
        nh.param<string>("lio_sam/odomTopic", odom_topic, "odometry");
        nh.param<bool>("experiment/eval_tracking", eval_tracking, false);
        nh.param<bool>("experiment/eval_slam", eval_slam, false);
        nh.param<float>("experiment/lidar_scope", lidar_scope, 40.0);
        nh.param<int>("experiment/max_out_cnt", max_out_cnt, 10);
        nh.param<int>("experiment/min_pt_cnt", min_pt_cnt, 10);

        t_ego_lidar = tf::Transform(tf::createQuaternionFromYaw(0.0), tf::Vector3(-0.2, 0.0, 1.9));        
        
        // // tracking
        pub_trk_est_bbox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/exp/trk/est_bbox", 1);
        pub_trk_cloud = nh.advertise<sensor_msgs::PointCloud2>("/exp/trk/cloud", 1);
        sub_est_bbox = nh.subscribe<jsk_recognition_msgs::BoundingBoxArray>("/exp/est_bbox", 2000, &Experiment::estBoundingBoxCallback, this, ros::TransportHints().tcpNoDelay());
        sub_gt_bbox = nh.subscribe<derived_object_msgs::ObjectArray>("/carla/objects", 2000, &Experiment::objectCallback, this, ros::TransportHints().tcpNoDelay());
        pub_gt_bbox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/exp/gt_bbox", 1);
        pub_cmp_bbox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/exp/cmp_bbox", 1);

        // slam
        pub_gt_odom = nh.advertise<nav_msgs::Odometry>("/exp/gt_odom", 1);
        pub_estimated_odom = nh.advertise<nav_msgs::Odometry>("/exp/est_odom", 1);
        sub_estimated_odom = nh.subscribe<nav_msgs::Odometry>("/lio_sam/mapping/odometry", 2000, &Experiment::estimatedOdomCallback, this, ros::TransportHints().tcpNoDelay());
        
        // semantic
        sub_raw_cloud = nh.subscribe<sensor_msgs::PointCloud2>("/carla/ego_vehicle/semantic_lidar/", 2000, &Experiment::cloudCallback, this, ros::TransportHints().tcpNoDelay());
        sub_tf = nh.subscribe<tf2_msgs::TFMessage>("/tf", 2000, &Experiment::tfCallback, this, ros::TransportHints().tcpNoDelay());
        
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

    }
    void objectCallback(const derived_object_msgs::ObjectArray::ConstPtr &msg_in) {
        jsk_recognition_msgs::BoundingBoxArray bbox_array;
        bbox_array.header.stamp = msg_in->header.stamp;
        bbox_array.header.frame_id = "carla_map";
        for (size_t i = 0; i < msg_in->objects.size(); i++) {
            derived_object_msgs::Object obj = msg_in->objects[i];
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
        pcl::moveFromROSMsg(current_cloud, *carla_cloud);

        cout << "Time: " << msg_in->header.stamp.toSec() << endl;
        for (auto pt : carla_cloud->points) {
            // for (const auto bbox : cur_bbox_array.boxes) {
            // }
            if (pt.ObjIdx > 0) {
                // cout << pt.x << ";" << pt.y << ";" << pt.z << " idx: " << pt.ObjIdx << " tag: " << pt.ObjTag << endl;
            }
        }
    }

    void estBoundingBoxCallback(const jsk_recognition_msgs::BoundingBoxArray::ConstPtr &msg_in) {
        if (eval_tracking) {
            jsk_recognition_msgs::BoundingBoxArray cur_bbox = *msg_in;
            est_bbox_queue.push_back(cur_bbox);
            tracking_evaluate();
        }
    }

    void estimatedOdomCallback(const nav_msgs::Odometry::ConstPtr &msg_in) {
        if (eval_slam) {
            nav_msgs::Odometry cur_odom = *msg_in;
            est_odom_queue.push_back(cur_odom);
            slam_evaluate();
        }
    }

    void tracking_evaluate() {

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
        cout << "Found at: " << gt_odom.header.stamp.toSec() << endl;
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

        for (auto gt : gt_bbox.boxes) {
            float dist = sqrt(
                        (gt.pose.position.x) * (gt.pose.position.x) +
                        (gt.pose.position.y) * (gt.pose.position.y) +
                        (gt.pose.position.z) * (gt.pose.position.z)); 
                        
            std::shared_ptr<Track> cur_tracker(new Track(gt.label, (int)gt.value));

            auto it = track_table.find(gt.label);
            if (it != track_table.end()) // exist
                *cur_tracker = track_table[gt.label];

            if (dist <= lidar_scope && dist >= 1.5 && insideBBOX(current_cloud, gt)) { 
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
                    }
                }

                // one or more candidates inside gt_bbox
                cout << "gt: " << gt.pose.position.x << ";"<<gt.pose.position.y << ";"<<gt.pose.position.z << endl;
                if (candidates.size() >= 1) {
                    if (candidates.size() > 1) {
                        c_fragmentation++;
                        cout << "Fragmentation: " << c_fragmentation<< endl;
                    }
                    
                    // set id of highest iou bbox to gt_bbox's id
                    int max_idx = std::max_element(ious.begin(), ious.end()) - ious.begin();
                    
                    if (cur_tracker->pred_label != candidates[max_idx].label) { // id switch
                        c_id_switch++;
                        if (cur_tracker->trk_cnt > cur_tracker->max_trk_cnt) { // update max tracking cnt
                            cur_tracker->max_trk_cnt = cur_tracker->trk_cnt;
                        }
                        cur_tracker->trk_cnt = 0;
                        cout << "Switch" << endl;
                    } else {
                        cout << "Success" << endl;
                        cur_tracker->trk_cnt++;
                        if (cur_tracker->trk_cnt > cur_tracker->max_trk_cnt) { // update max tracking cnt
                            cur_tracker->max_trk_cnt = cur_tracker->trk_cnt;
                        }
                    }
                    cur_tracker->pred_label = candidates[max_idx].label;
                } else { // no est_bbox associated
                    cout << "Not associated" << endl;
                    c_false_neg++;
                }
                cur_tracker->trk_ratio = cur_tracker->max_trk_cnt / (float)cur_tracker->in_cnt;
                cur_tracker->print();
            } else { // outside tracking range
                cur_tracker->out_cnt++;
                if (cur_tracker->out_cnt > max_out_cnt) { // ~2 seconds -> reset
                    cur_tracker->reset();
                }                    
            }
            track_table[gt.label] = *cur_tracker;
        }
        // False positive

        // update overall metric
        stat.t_FN += c_false_neg;
        stat.t_FP += c_false_pos;
        stat.t_FRAG += c_fragmentation;
        stat.t_SWC += c_id_switch;
        stat.t_GT += c_gt_cnt;
        stat.FN.push_back(c_false_neg);
        stat.FP.push_back(c_false_pos);
        stat.FRAG.push_back(c_fragmentation);
        stat.SWC.push_back(c_id_switch);
        stat.GT.push_back(c_gt_cnt);
        stat.MOTA.push_back(1 - float(c_false_pos + c_false_neg + c_id_switch) / c_gt_cnt);
        cout << "FN: " << c_false_neg << " FP: " << c_false_pos << " Frag: " << c_fragmentation << " Switch: " << c_id_switch << " gt: " << c_gt_cnt << endl;
        cout << "MOTA: " << 1-(c_false_pos + c_false_neg + c_id_switch) / (float)c_gt_cnt << endl;
    
        if (pub_cmp_bbox.getNumSubscribers() != 0)
            pub_cmp_bbox.publish(inside_scope_bbox);
    }

    bool insideBBOX(sensor_msgs::PointCloud2 this_cloud, jsk_recognition_msgs::BoundingBox this_bbox) {
        // 2. convert current semantic point cloud to pcl
        pcl::PointCloud<PointCarla>::Ptr pcl_cloud(new pcl::PointCloud<PointCarla>());
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
                cnt++;
            }
            if (cnt >= min_pt_cnt) {
                return true;
            }
        }
        // cout << "Inside: " << endl;
        // cout << "Center: " << this_bbox.pose.position.x << " " << this_bbox.pose.position.y << " " << this_bbox.pose.position.z << " count: " << cnt << endl;
        return false;
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
        cout << "Found at: " << gt_odom.header.stamp.toSec() << endl;
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
