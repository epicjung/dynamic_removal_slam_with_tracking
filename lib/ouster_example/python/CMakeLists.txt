cmake_minimum_required(VERSION 3.1.0)

set(OUSTER_SDK_PATH "${CMAKE_CURRENT_LIST_DIR}/.." CACHE STRING "SDK source directory")
file(TO_CMAKE_PATH "${OUSTER_SDK_PATH}" OUSTER_SDK_PATH)
message(STATUS "Ouster SDK location: ${OUSTER_SDK_PATH}")
list(APPEND CMAKE_MODULE_PATH ${OUSTER_SDK_PATH}/cmake)

project(python-ouster-sdk)

# ==== Options ====
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W3 /wd4996)
  add_compile_definitions(NOMINMAX _USE_MATH_DEFINES WIN32_LEAN_AND_MEAN)
else()
  add_compile_options(-Wall -Wextra -Werror -Wno-error=deprecated-declarations)
endif()

option(BUILD_VIZ "Enabled for Python build" ON)
option(BUILD_PCAP "Enabled for Python build" ON)

# ==== Requirements ====
find_package(pybind11 2.0 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OusterSDK REQUIRED)

# CMAKE_LIBRARY_OUTPUT_DIRECTORY is set in setup.py to the root of the `ouster`
# namespace, but we have to provide per-target packages directories for each
# extension module here.
set(EXT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# Note: With multi-configuration generators (like for VS), CMake automatically
# appends build-configuration suffix to *_OUTPUT_DIRECTORY properties *unless*
# they contain a generator expression, so we use a noop: $<0:>
# https://cmake.org/cmake/help/latest/prop_tgt/LIBRARY_OUTPUT_DIRECTORY.html
pybind11_add_module(_client src/cpp/_client.cpp)
target_link_libraries(_client PRIVATE ouster_client)
target_include_directories(_client SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})
set_target_properties(_client PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE
  LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}/client/$<0:>)

pybind11_add_module(_pcap src/cpp/_pcap.cpp)
target_link_libraries(_pcap PRIVATE ouster_pcap)
target_include_directories(_pcap SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})
set_target_properties(_pcap PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE
  LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}/pcap/$<0:>)

pybind11_add_module(_viz src/cpp/_viz.cpp)
target_link_libraries(_viz PRIVATE ouster_client ouster_viz)
target_include_directories(_viz SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})
set_target_properties(_viz PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE
  LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}/sdk/$<0:>)
