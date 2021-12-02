cmake_minimum_required(VERSION 2.8.3)
project(experiment)
set(CMAKE_BUILD_TYPE "Release")

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)


find_package(catkin REQUIRED
    message_generation
    roscpp
    std_msgs
    nav_msgs
    tf
    geometry_msgs
    derived_object_msgs
    jsk_recognition_msgs
    pcl_conversions
    visualization_msgs
)

find_package(PCL REQUIRED QUIET)

catkin_package(
  INCLUDE_DIRS
  LIBRARIES
  CATKIN_DEPENDS roscpp std_msgs
)

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
 ${PCL_INCLUDE_DIRS}
 src
)

add_executable(exp_node src/experiment.cpp src/iou.cpp src/iou.h)
add_dependencies(exp_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(exp_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
