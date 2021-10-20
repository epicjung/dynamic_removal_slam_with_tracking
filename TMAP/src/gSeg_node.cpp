#include <iostream>
#define PCL_NO_PRECOMPILE

#include "TMAP/patchwork_obstacle.hpp"

boost::shared_ptr<Patchwork_M> patchwork_m;

ros::Publisher pub_nongroundpc;
ros::Publisher pub_groundpc;
ros::Publisher pub_obstaclepc;
ros::Publisher pub_LIG_Node;
ros::Publisher pub_gng;

std::string node_topic_;
std::string ground_filter_algorithm_;
std::string ptCloud_type_;


int main(int argc, char **argv)
{
    ros::init(argc, argv, "patchwork");

    ros::NodeHandle nh;

    // set parameters
    nh.param<std::string>("/nodegen/node_topic", node_topic_, "/lig_node/pre_processing_node");
    nh.param<std::string>("/nodegen/ptCloud_type", ptCloud_type_, "None");
    nh.param<std::string>("/gSeg_algorithm", ground_filter_algorithm_, "patchwork");

    ROS_INFO_STREAM("TrvNode  Topic: "+ node_topic_);
    ROS_INFO_STREAM("Algorithm Name: "+ ground_filter_algorithm_);
    ROS_INFO_STREAM("PtCloud  Align: "+ ptCloud_type_);

    if(ground_filter_algorithm_ == "patchwork_m"){
        patchwork_m.reset(new Patchwork_M(&nh));
    } else{
        ROS_ERROR("Please set the algorithm name correctly");
    }

    ros::spin();
    return 0;
}
