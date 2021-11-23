#include <ros/ros.h>
#include <derived_object_msgs/ObjectArray.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <fstream>
#include <cstdio>
#include <experimental/filesystem>
#include <signal.h>
using namespace std;


class Experiment
{
    public:

        tf::Transform t_ego_lidar;
        tf::StampedTransform t_world_start;
        
        string SAVE_DIR;
        
        // string SAVE_DIR = "/home/euigon/experiment/traj_result/";
        string gt_file = "stamped_groundtruth.txt";
        string est_file = "stamped_traj_estimate.txt";
        vector<geometry_msgs::PoseStamped> estimated_poses;
        vector<geometry_msgs::PoseStamped> groundtruth_poses;
        bool first_odom_in;

        ros::Publisher pub_gt_bbox;
        ros::Publisher pub_gt_odom;
        ros::Subscriber sub_estimated_odom;
        ros::Subscriber sub_gt_odom;
        ros::Subscriber sub_objects;

    Experiment() {

        ros::NodeHandle nh("~");

        first_odom_in = false;

        nh.param<string>("save_dir", SAVE_DIR, "/home/euigon/experiment/traj_result/");
        t_ego_lidar = tf::Transform(tf::createQuaternionFromYaw(0.0), tf::Vector3(-0.2, 0.0, 1.9));        
        pub_gt_bbox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/tracking/gt_bbox", 1);
        pub_gt_odom = nh.advertise<nav_msgs::Odometry>("/exp/gt_odom", 1);
        sub_estimated_odom = nh.subscribe<nav_msgs::Odometry>("/lio_sam/mapping/odometry", 2000, &Experiment::estimatedOdomCallback, this, ros::TransportHints().tcpNoDelay());
        sub_gt_odom = nh.subscribe<nav_msgs::Odometry>("/carla/ego_vehicle/odometry", 2000, &Experiment::gtOdomCallback, this, ros::TransportHints().tcpNoDelay());
        sub_objects = nh.subscribe<derived_object_msgs::ObjectArray>("/carla/objects", 2000, &Experiment::objectCallback, this, ros::TransportHints().tcpNoDelay());

        ofstream gt_output(SAVE_DIR + gt_file);
        ofstream est_output(SAVE_DIR + est_file);

        gt_output.close();
        est_output.close();
    }

    ~Experiment() {
        std::cout<<"\033[1;32m"<< "You pressed Ctrl+C..." <<"\033[0m"<<std::endl;
        std::cout<<"\033[1;32m"<< "Saving to " << SAVE_DIR << "\033[0m"<<std::endl;

        std::ofstream est_output(SAVE_DIR + est_file, std::ios::app);
        std::ofstream gt_output(SAVE_DIR + gt_file, std::ios::app);

        for (geometry_msgs::PoseStamped pose_it : groundtruth_poses){

            std_msgs::Header header = pose_it.header;
            geometry_msgs::Point tl = pose_it.pose.position;
            geometry_msgs::Quaternion qt = pose_it.pose.orientation;

            gt_output<<header.stamp<<" "<<tl.x<<" "<<tl.y<<" "<<tl.z<< " "<<qt.x <<" "<< qt.y<<" "<<qt.z<<" "<<qt.w;
            gt_output<<std::endl;
        }

        for (geometry_msgs::PoseStamped pose_it : estimated_poses){

            std_msgs::Header header = pose_it.header;
            geometry_msgs::Point tl = pose_it.pose.position;
            geometry_msgs::Quaternion qt = pose_it.pose.orientation;

            est_output<<header.stamp<<" "<<tl.x<<" "<<tl.y<<" "<<tl.z<< " "<<qt.x <<" "<< qt.y<<" "<<qt.z<<" "<<qt.w;
            est_output<<std::endl;
        }

        gt_output.close();
        est_output.close();

        std::cout<<"\033[1;32m"<< "Done..." <<"\033[0m"<<std::endl;

        exit(1);
    }

    void objectCallback(const derived_object_msgs::ObjectArray::ConstPtr &msg_in) {
        // jsk_recognition_msgs::BoundingBoxArray bbox_array;
        // bbox_array.header.stamp = timestamp;
        // bbox_array.header.frame_id = this_frame;
        // std::vector<Cluster> cluster_map = map.getMap();
        // for (size_t i = 0; i < cluster_map.size(); ++i)
        // {
        //     jsk_recognition_msgs::BoundingBox bbox = cluster_map[i].bbox;
        //     // if (bbox.value >= 0)
        //     // {
        //         bbox.header.stamp = timestamp;
        //         bbox.header.frame_id = this_frame;
        //         bbox_array.boxes.push_back(bbox);  
        //     // }
        // }
        // pub_gt_bbox.publish(bbox_array);
    }

    void estimatedOdomCallback(const nav_msgs::Odometry::ConstPtr &msg_in) {
        
        static tf::TransformListener listener;

        if (!first_odom_in) {
            try{
                listener.waitForTransform("map", "ego_vehicle", msg_in->header.stamp, ros::Duration(0.01));
                listener.lookupTransform("map", "ego_vehicle", msg_in->header.stamp, t_world_start);
                first_odom_in = true;
            } 
            catch (tf::TransformException ex){
                ROS_ERROR("no imu tf");
                return;
            }
        }

        if (first_odom_in) {
            tf::Transform t_odom_baselink;
            tf::poseMsgToTF(msg_in->pose.pose, t_odom_baselink);
            t_world_start.setRotation(tf::createQuaternionFromYaw(0.0));
            tf::Transform t_world_lidar = t_world_start * t_ego_lidar * t_odom_baselink;
            printf("x: %f y: %f z: %f\n", t_world_lidar.getOrigin().x(),  t_world_lidar.getOrigin().y(),  t_world_lidar.getOrigin().z());
            geometry_msgs::PoseStamped estimated;
            estimated.header = msg_in->header;
            estimated.pose.position.x = t_world_lidar.getOrigin().x();
            estimated.pose.position.y = t_world_lidar.getOrigin().y();
            estimated.pose.position.z = t_world_lidar.getOrigin().z();
            estimated.pose.orientation.x = t_world_lidar.getRotation().x();
            estimated.pose.orientation.y = t_world_lidar.getRotation().y();
            estimated.pose.orientation.z = t_world_lidar.getRotation().z();
            estimated.pose.orientation.w = t_world_lidar.getRotation().w();
            estimated_poses.push_back(estimated);
            printf("Estimated poses: %d\n", estimated_poses.size());
        }
    }

    void gtOdomCallback(const nav_msgs::Odometry::ConstPtr &msg_in) {

        static tf::TransformBroadcaster tf_world_to_ego;
        static double prev_time = msg_in->header.stamp.toSec();
        double interval = msg_in->header.stamp.toSec() - prev_time;

        tf::Transform t_world_ego;
        tf::poseMsgToTF(msg_in->pose.pose, t_world_ego);

        if (!first_odom_in) {
            tf::StampedTransform stamped_world_ego = tf::StampedTransform(t_world_ego, msg_in->header.stamp, "map", "ego_vehicle");
            tf_world_to_ego.sendTransform(stamped_world_ego);
        }
        
        tf::Transform t_world_lidar = t_world_ego * t_ego_lidar;

        if (first_odom_in) {
            nav_msgs::Odometry lidar_odom;
            tf::Transform t_odom_gt = t_ego_lidar.inverse() * t_world_start.inverse() * t_world_lidar;
            lidar_odom.header.frame_id = "odom";
            lidar_odom.header.stamp = msg_in->header.stamp;
            lidar_odom.pose.pose.position.x = t_odom_gt.getOrigin().x();
            lidar_odom.pose.pose.position.y = t_odom_gt.getOrigin().y();
            lidar_odom.pose.pose.position.z = t_odom_gt.getOrigin().z();
            lidar_odom.pose.pose.orientation.x = t_odom_gt.getRotation().x();
            lidar_odom.pose.pose.orientation.y = t_odom_gt.getRotation().y();
            lidar_odom.pose.pose.orientation.z = t_odom_gt.getRotation().z();
            lidar_odom.pose.pose.orientation.w = t_odom_gt.getRotation().w();            
            pub_gt_odom.publish(lidar_odom);
        }

        if (interval > 0.1) {
            geometry_msgs::PoseStamped ground_truth;
            ground_truth.header = msg_in->header;
            ground_truth.pose.position.x = t_world_lidar.getOrigin().x();
            ground_truth.pose.position.y = t_world_lidar.getOrigin().y();
            ground_truth.pose.position.z = t_world_lidar.getOrigin().z();
            ground_truth.pose.orientation.x = t_world_lidar.getRotation().x();
            ground_truth.pose.orientation.y = t_world_lidar.getRotation().y();
            ground_truth.pose.orientation.z = t_world_lidar.getRotation().z();
            ground_truth.pose.orientation.w = t_world_lidar.getRotation().w();
            groundtruth_poses.push_back(ground_truth);
            prev_time = msg_in->header.stamp.toSec();
            printf("Ground-truth poses: %d\n", groundtruth_poses.size());
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "experiment");

    Experiment exp;

    ROS_INFO("\033[1;32m----> experiment started.\033[0m");

    ros::spin();

    return 0;
}
