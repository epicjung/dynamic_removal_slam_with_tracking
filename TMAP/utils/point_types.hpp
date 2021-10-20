#pragma once

#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>

struct EIGEN_ALIGN16 PointXYZIRT {
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRT,
    (float, x, x) 
    (float, y, y) 
    (float, z, z) 
    (float, intensity, intensity)
    (uint16_t, ring, ring) 
    (float, time, time)
)

namespace velodyne_ptype{
    struct EIGEN_ALIGN16 PointXYZIRT {
        PCL_ADD_POINT4D
        PCL_ADD_INTENSITY;
        uint16_t ring;
        float time;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}

POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ptype::PointXYZIRT,
    (float, x, x) 
    (float, y, y) 
    (float, z, z) 
    (float, intensity, intensity)
    (uint16_t, ring, ring) 
    (float, time, time)
)

namespace ouster_ptype {

    struct EIGEN_ALIGN16 PointOS0 {
        PCL_ADD_POINT4D;
        float intensity;
        uint32_t t;
        uint16_t reflectivity;
        uint8_t ring;
        uint16_t ambient;
        uint32_t range;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    struct EIGEN_ALIGN16 PointOS1 {
        PCL_ADD_POINT4D;
        float intensity;
        uint32_t t;
        uint16_t reflectivity;
        uint8_t ring;
        uint16_t noise;
        uint32_t range;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        static inline PointOS1 make(float x, float y, float z, float intensity,
                                    uint32_t t, uint16_t reflectivity, uint8_t ring,
                                    uint16_t noise, uint32_t range) {
            return {x, y, z, 0.0, intensity, t, reflectivity, ring, noise, range};
        }
    };
}

POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ptype::PointOS0,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    // use std::uint32_t to avoid conflicting with pcl::uint32_t
    (std::uint32_t, t, t)
    (std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring)
    (std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range)
)

POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ptype::PointOS1,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (uint32_t, t, t)
    (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring)
    (uint16_t, noise, noise)
    (uint32_t, range, range)
)

