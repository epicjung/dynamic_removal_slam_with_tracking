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

#include <thread>
#include <deque>
#include <mutex>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <experimental/filesystem>
#include <signal.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

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

class Experiment
{
    public:

        tf::Transform t_ego_lidar;
        tf::StampedTransform t_world_start;
        
        // params
        string save_dir;
        string odom_topic;

        // string SAVE_DIR = "/home/euigon/experiment/traj_result/";
        string gt_file = "stamped_groundtruth.txt";
        string est_file = "stamped_traj_estimate.txt";
        vector<nav_msgs::Odometry> estimated_poses;
        vector<nav_msgs::Odometry> groundtruth_poses;
        bool first_odom_in;

        // Tracking
        ros::Publisher pub_trk_gt_bbox;
        ros::Publisher pub_trk_gt_odom;
        ros::Publisher pub_trk_est_bbox;
        ros::Publisher pub_trk_est_odom;
        ros::Subscriber sub_gt_bbox;
        ros::Subscriber sub_est_bbox;
        ros::Subscriber sub_odom_baselink;
        ros::Publisher pub_gt_bbox;
        ros::Publisher pub_trk_cloud;

        // SLAM
        ros::Publisher pub_gt_odom;
        ros::Publisher pub_estimated_odom;
        ros::Subscriber sub_estimated_odom;
        ros::Subscriber sub_gt_odom;
        
        // semantic
        ros::Subscriber sub_tf;

        // Raw
        ros::Subscriber sub_raw_cloud;

        mutex gt_obj_mtx;
        mutex est_obj_mtx;
        mutex gt_odom_mtx;
        mutex est_odom_mtx;
        mutex imu_odom_mtx;
        deque<jsk_recognition_msgs::BoundingBoxArray> gt_obj_queue;
        deque<jsk_recognition_msgs::BoundingBoxArray> est_obj_queue;
        deque<nav_msgs::Odometry> gt_odom_queue;
        deque<nav_msgs::Odometry> gt_trk_odom_queue;
        deque<nav_msgs::Odometry> est_odom_queue;
        deque<nav_msgs::Odometry> imu_odom_queue;
        deque<geometry_msgs::TransformStamped> lidar_odom_queue;
        deque<sensor_msgs::PointCloud2> cloud_queue;
        tf::Transform t_map_odom;

    Experiment() {

        ros::NodeHandle nh("~");

        first_odom_in = false;

        nh.param<string>("save_dir", save_dir, "/home/euigon/experiment/traj_result/");
        nh.param<string>("lio_sam/odomTopic", odom_topic, "odometry");

        t_ego_lidar = tf::Transform(tf::createQuaternionFromYaw(0.0), tf::Vector3(-0.2, 0.0, 1.9));        
        
        // tracking
        pub_trk_gt_bbox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/exp/trk/gt_bbox", 1);
        pub_trk_gt_odom = nh.advertise<nav_msgs::Odometry>("/exp/trk/gt_odom", 1);
        pub_trk_est_bbox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/exp/trk/est_bbox", 1);
        pub_trk_est_odom = nh.advertise<nav_msgs::Odometry>("/exp/trk/est_odom", 1);
        pub_trk_cloud = nh.advertise<sensor_msgs::PointCloud2>("/exp/trk/cloud", 1);
        pub_gt_bbox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/exp/gt_bbox", 1);
        sub_gt_bbox = nh.subscribe<derived_object_msgs::ObjectArray>("/carla/objects", 2000, &Experiment::objectCallback, this, ros::TransportHints().tcpNoDelay());
        sub_est_bbox = nh.subscribe<jsk_recognition_msgs::BoundingBoxArray>("/exp/est_bbox", 2000, &Experiment::estBoundingBoxCallback, this, ros::TransportHints().tcpNoDelay());
        sub_odom_baselink = nh.subscribe<nav_msgs::Odometry>(odom_topic, 2000, &Experiment::imuOdomCallback, this, ros::TransportHints().tcpNoDelay());

        // slam
        pub_gt_odom = nh.advertise<nav_msgs::Odometry>("/exp/gt_odom", 1);
        pub_estimated_odom = nh.advertise<nav_msgs::Odometry>("/exp/est_odom", 1);
        sub_estimated_odom = nh.subscribe<nav_msgs::Odometry>("/lio_sam/mapping/odometry", 2000, &Experiment::estimatedOdomCallback, this, ros::TransportHints().tcpNoDelay());
        sub_gt_odom = nh.subscribe<nav_msgs::Odometry>("/carla/ego_vehicle/odometry", 2000, &Experiment::gtOdomCallback, this, ros::TransportHints().tcpNoDelay());
        
        // semantic
        sub_raw_cloud = nh.subscribe<sensor_msgs::PointCloud2>("/carla/ego_vehicle/semantic_lidar/lidar/point_cloud", 2000, &Experiment::cloudCallback, this, ros::TransportHints().tcpNoDelay());
        sub_tf = nh.subscribe<tf2_msgs::TFMessage>("/tf", 2000, &Experiment::tfCallback, this, ros::TransportHints().tcpNoDelay());
        ofstream gt_output(save_dir + gt_file);
        ofstream est_output(save_dir + est_file);

        gt_output.close();
        est_output.close();

    }

    ~Experiment() {
        std::cout<<"\033[1;32m"<< "You pressed Ctrl+C..." <<"\033[0m"<<std::endl;
        std::cout<<"\033[1;32m"<< "Saving to " << save_dir << "\033[0m"<<std::endl;

        std::ofstream est_output(save_dir + est_file, std::ios::app);
        std::ofstream gt_output(save_dir + gt_file, std::ios::app);

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

    void tfCallback(const tf2_msgs::TFMessageConstPtr &msg_in) {
        for (const auto ts_msg : msg_in->transforms) {
            if (ts_msg.header.frame_id == "map" && ts_msg.child_frame_id == "ego_vehicle/semantic_lidar/lidar") {
                lidar_odom_queue.push_back(ts_msg);
            }
        }
    }

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg_in) {
        cloud_queue.push_back(*msg_in);

        sensor_msgs::PointCloud2 current_cloud = std::move(*msg_in);
        pcl::PointCloud<PointCarla>::Ptr carla_cloud(new pcl::PointCloud<PointCarla>());
        pcl::moveFromROSMsg(current_cloud, *carla_cloud);
        cout << "time: " << msg_in->header.stamp.toSec() << endl;
        for (auto pt : carla_cloud->points) {
            cout << pt.x << " " << pt.y << " " << pt.z << " " << pt.ObjIdx << " " << pt.ObjTag << endl;
        }
    }

    void imuOdomCallback(const nav_msgs::Odometry::ConstPtr &msg_in) {
        imu_odom_mtx.lock();
        imu_odom_queue.push_back(*msg_in);
        imu_odom_mtx.unlock();
    }

    void estBoundingBoxCallback(const jsk_recognition_msgs::BoundingBoxArray::ConstPtr &msg_in) {
        est_obj_mtx.lock();
        est_obj_queue.push_back(*msg_in);
        est_obj_mtx.unlock();
        tracking_evaluate();
    }

    void objectCallback(const derived_object_msgs::ObjectArray::ConstPtr &msg_in) {
        jsk_recognition_msgs::BoundingBoxArray bbox_array;
        bbox_array.header = msg_in->header;
        for (size_t i = 0; i < msg_in->objects.size(); i++) {
            derived_object_msgs::Object obj = msg_in->objects[i];
            jsk_recognition_msgs::BoundingBox bbox;
            bbox.label = obj.id;
            bbox.header = msg_in->header;
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
            if (obj.classification != 4) {
                bbox.pose.position.z += obj.shape.dimensions[2] / 2.0;
            }
            bbox_array.boxes.push_back(bbox);
        }
        pub_gt_bbox.publish(bbox_array);
        gt_obj_mtx.lock();
        gt_obj_queue.push_back(bbox_array);
        gt_obj_mtx.unlock();
    }

    void estimatedOdomCallback(const nav_msgs::Odometry::ConstPtr &msg_in) {
        est_odom_mtx.lock();
        est_odom_queue.push_back(*msg_in);
        est_odom_mtx.unlock();
        slam_evaluate();
    }

    void gtOdomCallback(const nav_msgs::Odometry::ConstPtr &msg_in) {
        gt_odom_mtx.lock();
        gt_odom_queue.push_back(*msg_in);
        gt_trk_odom_queue.push_back(*msg_in);
        gt_odom_mtx.unlock();
    }

    void tracking_evaluate() {
        if (!first_odom_in)
            return;

        if (est_obj_queue.empty() || gt_obj_queue.empty() || cloud_queue.empty()) 
            return;
         
        jsk_recognition_msgs::BoundingBoxArray cur_bbox_array;   
        cur_bbox_array = est_obj_queue.front();
        est_obj_queue.pop_front();

        cur_bbox_array.header.frame_id = "old_base_link";
        for (size_t i = 0; i < cur_bbox_array.boxes.size(); i++) {
            cur_bbox_array.boxes[i].header.frame_id = "old_base_link";
        }

        // 0. Look for cloud
        while (cloud_queue.front().header.stamp < cur_bbox_array.header.stamp) {
            cloud_queue.pop_front();
        }
        if (cloud_queue.empty())
            return;
        assert(cloud_queue.front().header.stamp.toSec() == cur_box_array.header.stamp.toSec());

        sensor_msgs::PointCloud2 cur_cloud = cloud_queue.front();
        cur_cloud.header.frame_id = "old_base_link";
        pub_trk_cloud.publish(cur_cloud);

        // 1. Look for "carla_map" -> "ego" transform
        tf::Transform t_map_ego;
        while (gt_trk_odom_queue.front().header.stamp < cur_bbox_array.header.stamp) {
            gt_trk_odom_queue.pop_front();
        }
        if (gt_trk_odom_queue.empty())
            return;
        
        assert(gt_trk_odom_queue.front().header.stamp.toSec() == cur_box_array.header.stamp.toSec());

        nav_msgs::Odometry cur_gt_odom = gt_trk_odom_queue.front();
        tf::poseMsgToTF(cur_gt_odom.pose.pose, t_map_ego);
        cur_gt_odom.header.frame_id = "carla_map";

        // 2. find the ground-truth bounding boxes
        while (gt_obj_queue.front().header.stamp < cur_bbox_array.header.stamp){
            gt_obj_queue.pop_front();                
        }
        if (gt_obj_queue.empty())
            return;

        assert(gt_obj_queue.front().header.stamp.toSec() == cur_box_array.header.stamp.toSec());
        

        // 3. Transform ground-truth boxes to "baselink" frame
        jsk_recognition_msgs::BoundingBoxArray gt_bbox_array = gt_obj_queue.front();
        jsk_recognition_msgs::BoundingBoxArray out_gt_bbox_array;
        out_gt_bbox_array.header.stamp = gt_bbox_array.header.stamp;
        out_gt_bbox_array.header.frame_id = "old_base_link";
        cout << "Tracking found at: " << gt_bbox_array.header.stamp.toSec() << endl;
    
        for (size_t i = 0; i < gt_bbox_array.boxes.size(); i++) {
            jsk_recognition_msgs::BoundingBox bbox = gt_bbox_array.boxes[i];

            tf::Transform t_map_obj;
            tf::poseMsgToTF(bbox.pose, t_map_obj);
            tf::Transform t_baselink_obj = t_ego_lidar.inverse() * t_map_ego.inverse() * t_map_obj;

            if (bbox.label == 304) { // ego
                cout << "map_ego: " << t_map_ego.getOrigin().x() << " " << t_map_ego.getOrigin().y() << " " << t_map_ego.getOrigin().z() << endl;
                cout << "map_obj: " << t_map_obj.getOrigin().x() << " " << t_map_obj.getOrigin().y() << " " << t_map_obj.getOrigin().z() << endl;
                cout << "bl_obj: " << t_baselink_obj.getOrigin().x() << " " << t_baselink_obj.getOrigin().y() << " " << t_baselink_obj.getOrigin().z() << endl;
            }

            bbox.header.frame_id = "old_base_link";

            // set pose
            bbox.pose.position.x = t_baselink_obj.getOrigin().x();
            bbox.pose.position.y = t_baselink_obj.getOrigin().y();
            bbox.pose.position.z = t_baselink_obj.getOrigin().z();
            bbox.pose.orientation.x = t_baselink_obj.getRotation().x();
            bbox.pose.orientation.y = t_baselink_obj.getRotation().y();
            bbox.pose.orientation.z = t_baselink_obj.getRotation().z();
            bbox.pose.orientation.w = t_baselink_obj.getRotation().w();
            out_gt_bbox_array.boxes.push_back(bbox);
        }

        cout << "Publish current status: " << cur_bbox_array.header.stamp.toSec() << endl;
        pub_trk_est_bbox.publish(cur_bbox_array);
        pub_trk_gt_odom.publish(cur_gt_odom);
        pub_trk_gt_bbox.publish(out_gt_bbox_array);
        cout << "End of current status" << endl;
    }

    // void tracking_evaluate() {
    //     if (!first_odom_in)
    //         return;

    //     if (est_obj_queue.empty() || gt_obj_queue.empty()) 
    //         return;
         
    //     derived_object_msgs::ObjectArray gt_obj_array;
    //     jsk_recognition_msgs::BoundingBoxArray cur_obj_array;   
    //     cur_obj_array = est_obj_queue.front();
    //     est_obj_queue.pop_front();
        
    //     // 1. Look for "odom" -> "baselink" transform
    //     while (imu_odom_queue.front().header.stamp < cur_obj_array.header.stamp) {
    //         imu_odom_queue.pop_front();
    //     }
    //     if (imu_odom_queue.empty())
    //         return;

    //     cout << "Est first: " << cur_obj_array.header.stamp.toSec() << endl;
    //     assert(imu_odom_queue.front().header.stamp == cur_obj_array.header.stamp.toSec());
    //     nav_msgs::Odometry imu_odom = imu_odom_queue.front();
    //     pub_tracking_odom.publish(imu_odom);
    //     tf::Transform t_odom_baselink;
    //     tf::poseMsgToTF(imu_odom_queue.front().pose.pose, t_odom_baselink);

    //     // 2. find the ground-truth bounding boxes
    //     while (gt_obj_queue.front().header.stamp < cur_obj_array.header.stamp){
    //         gt_obj_queue.pop_front();                
    //     }
    //     if (gt_obj_queue.empty())
    //         return;

    //     // 3. Transform ground-truth boxes to "baselink" frame
    //     gt_obj_array = gt_obj_queue.front();
    //     cout << "Tracking found at: " << gt_obj_array.header.stamp.toSec() << endl;
    //     tf::Transform t_baselink_map = (t_map_odom * t_odom_baselink).inverse();
    //     jsk_recognition_msgs::BoundingBoxArray bbox_array;
    //     bbox_array.header.stamp = gt_obj_array.header.stamp;
    //     bbox_array.header.frame_id = "base_link";
    //     for (size_t i = 0; i < gt_obj_array.objects.size(); i++) {
    //         derived_object_msgs::Object obj = gt_obj_array.objects[i];
    //         jsk_recognition_msgs::BoundingBox bbox;

    //         tf::Transform t_map_obj;
    //         tf::poseMsgToTF(obj.pose, t_map_obj);
    //         tf::Transform t_baselink_obj = t_baselink_map * t_map_obj;

    //         bbox.label = obj.id;
    //         bbox.header.stamp = gt_obj_array.header.stamp;
    //         bbox.header.frame_id = "base_link";

    //         // set dimension
    //         if (obj.shape.dimensions.size() == 3) { // bounding box
    //             bbox.dimensions.x = obj.shape.dimensions[0];
    //             bbox.dimensions.y = obj.shape.dimensions[1];
    //             bbox.dimensions.z = obj.shape.dimensions[2];
    //         } else {
    //             cout << "Not a bounding-box" << endl;
    //         }

    //         // set pose
    //         bbox.pose.position.x = t_baselink_obj.getOrigin().x();
    //         bbox.pose.position.y = t_baselink_obj.getOrigin().y();
    //         bbox.pose.position.z = t_baselink_obj.getOrigin().z();
    //         bbox.pose.orientation.x = t_baselink_obj.getRotation().x();
    //         bbox.pose.orientation.y = t_baselink_obj.getRotation().y();
    //         bbox.pose.orientation.z = t_baselink_obj.getRotation().z();
    //         bbox.pose.orientation.w = t_baselink_obj.getRotation().w();
    //         bbox_array.boxes.push_back(bbox);
    //     }
    //     pub_gt_bbox.publish(bbox_array);
    // }

    void slam_evaluate() {

        if (est_odom_queue.empty() || gt_odom_queue.empty()) 
            return;
        
        // Estimated pose
        nav_msgs::Odometry cur_odom_lidar;
        nav_msgs::Odometry gt_odom;                 
        cur_odom_lidar = est_odom_queue.front();
        est_odom_queue.pop_front();
        
        // Find the gt pose for the first estimated pose
        gt_odom_mtx.lock();
        while (gt_odom_queue.front().header.stamp < cur_odom_lidar.header.stamp){
            gt_odom_queue.pop_front();                
        }
        gt_odom_mtx.unlock();

        if (gt_odom_queue.empty())
            return;

        gt_odom = gt_odom_queue.front();
        cout << "Found at: " << gt_odom.header.stamp.toSec() << endl;
        assert(gt_odom.header.stamp.toSec() == cur_odom_lidar.header.stamp.toSec());

        tf::Transform t_map_ego;
        tf::poseMsgToTF(gt_odom.pose.pose, t_map_ego);
        tf::Transform t_map_lidar = t_map_ego * t_ego_lidar;

        // find map to odom transformation
        if (!first_odom_in) {
            t_map_ego.setRotation(tf::createQuaternionFromYaw(0.0));
            t_map_odom = t_map_ego * t_ego_lidar;
            first_odom_in = true;
        }

        // send odom to map (permanent)
        static tf::TransformBroadcaster tf_broadcaster;
        tf::StampedTransform stamped_odom_map = tf::StampedTransform(t_map_odom.inverse(), gt_odom.header.stamp, "odom", "carla_map");
        tf_broadcaster.sendTransform(stamped_odom_map);

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
