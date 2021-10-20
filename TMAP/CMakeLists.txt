cmake_minimum_required(VERSION 3.0.2)
project(TMAP)

add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE "Release")

# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslaunch
  pcl_conversions
  pcl_ros
  cv_bridge
  message_generation
  geometry_msgs
  sensor_msgs
  std_msgs
  tMap_msgs
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES
 CATKIN_DEPENDS roscpp rospy std_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  src
  utils
)

link_directories(
    ${PCL_LIBRARY_DIRS}
    ${OpenCV_LIBRARY_DIRS}
)

set(algorithm_src
    utils/ptcloud_handle.hpp
    utils/point_types.hpp
)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# # patchwork
# add_executable(gSeg_node src/gSeg_node.cpp ${algorithm_src})
# target_link_libraries(gSeg_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
# add_dependencies(gSeg_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

install(DIRECTORY include/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN ".svn" EXCLUDE)