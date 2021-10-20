#include "utility.h"
#include "lio_sam/cloud_info.h"
#include "imm_gmphd_filter.h"
#include "cluster.h"
#include "TMAP/patchwork_obstacle.hpp"

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

struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;

    ros::Subscriber subLaserCloud;
    ros::Publisher  pubLaserCloud;
    
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;

    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag;
    cv::Mat rangeMat;

    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lio_sam::cloud_info cloudInfo;
    double timeScanCur;
    double timeScanEnd;
    std_msgs::Header cloudHeader;

    boost::shared_ptr<Patchwork_M> m_patchwork;
    ClusterMap meas_map;
    ClusterMap tracker_map;
    ros::Publisher pub_meas_cluster;
    ros::Publisher pub_tracker_cluster;
    ros::Publisher pub_meas_centroid;
    ros::Publisher pub_tracker_centroid;
    ros::Publisher pub_bbox;
    ros::Publisher pub_meas_cluster_info;
    ros::Publisher pub_tracker_cluster_info;
    ros::Publisher pub_ground_cloud;
    ros::Publisher pub_nonground_cloud;
    ros::Publisher pub_nonground_ds_cloud;
    bool tracker_flag;
    
    gmphd::IMMGMPHD<2> target_tracker;

public:
    ImageProjection():
    deskewFlag(0)
    {
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());

        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("lio_sam/deskew/cloud_deskewed", 1);
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/deskew/cloud_info", 1);

        pub_meas_cluster         = nh.advertise<sensor_msgs::PointCloud2>("/tracking/lidar/meas_cluster", 1);
        pub_tracker_cluster         = nh.advertise<sensor_msgs::PointCloud2>("/tracking/lidar/track_cluster", 1);
        pub_meas_centroid        = nh.advertise<sensor_msgs::PointCloud2>("/tracking/lidar/meas_centroid", 1);
        pub_tracker_centroid        = nh.advertise<sensor_msgs::PointCloud2>("/tracking/lidar/tracker_centroid", 1);

        pub_bbox            = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/tracking/lidar/bbox", 1);
        pub_meas_cluster_info    = nh.advertise<visualization_msgs::MarkerArray>("/tracking/lidar/meas_cluster_info", 1);
        pub_tracker_cluster_info    = nh.advertise<visualization_msgs::MarkerArray>("/tracking/lidar/tracker_cluster_info", 1);

        pub_ground_cloud = nh.advertise<sensor_msgs::PointCloud2>("/segmentation/ground", 1);
        pub_nonground_cloud = nh.advertise<sensor_msgs::PointCloud2>("/segmentation/nonground", 1);
        pub_nonground_ds_cloud = nh.advertise<sensor_msgs::PointCloud2>("/segmentation/nonground_ds", 1);
        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        m_patchwork.reset(new Patchwork_M(&nh));

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }

        tracker_flag = false;
        meas_map.init(MEASUREMENT, maxR, maxZ, clusteringTolerance, minClusterSize, maxClusterSize);
        meas_map.setFittingParameters(minHeight, maxHeight, maxArea, maxRatio, minDensity);
        tracker_map.init(TRACKER, maxR, maxZ, 0.0, 0, 0);
    }

    ~ImageProjection(){}

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);

        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << 
        //       ", y: " << thisImu.linear_acceleration.y << 
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << 
        //       ", y: " << thisImu.angular_velocity.y << 
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        if (!cachePointCloud(laserCloudMsg))
            return;

        if (!deskewInfo())
            return;

        projectPointCloud();

        cloudExtraction();

        tracking();

        publishClouds();

        resetParameters();
    }

    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)
            return false;

        // convert cloud
        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();
        if (sensor == SensorType::VELODYNE)
        {
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
        }
        else if (sensor == SensorType::OUSTER)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else
        {
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            ros::shutdown();
        }

        // get timestamp
        cloudHeader = currentCloudMsg.header;
        timeScanCur = cloudHeader.stamp.toSec();
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

        // check dense flag
        if (laserCloudIn->is_dense == false)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // check ring channel
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }

        // check point time
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }

        imuDeskewInfo();

        odomDeskewInfo();

        return true;
    }

    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;

        while (!imuQueue.empty())
        {
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();

            // get roll, pitch, and yaw estimation for this scan
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);

            if (currentImuTime > timeScanEnd + 0.01)
                break;

            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;

        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true;
    }

    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;

        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // get start odometry at the beinning of the scan
        nav_msgs::Odometry startOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }

        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;

        cloudInfo.odomAvailable = true;

        // get end odometry at the end of the scan
        odomDeskewFlag = false;

        if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
            return;

        nav_msgs::Odometry endOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanEnd)
                continue;
            else
                break;
        }

        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }

    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

        if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
            return;

        float ratio = relTime / (timeScanEnd - timeScanCur);

        *posXCur = ratio * odomIncreX;
        *posYCur = ratio * odomIncreY;
        *posZCur = ratio * odomIncreZ;
    }

    PointType deskewPoint(PointType *point, double relTime)
    {
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;

        double pointTime = timeScanCur + relTime;

        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // transform points to start
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            float range = pointDistance(thisPoint);
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;

            int rowIdn = laserCloudIn->points[i].ring;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            if (rowIdn % downsampleRate != 0)
                continue;

            float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

            static float ang_res_x = 360.0/float(Horizon_SCAN);
            int columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);

            rangeMat.at<float>(rowIdn, columnIdn) = range;

            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }

    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry
        for (int i = 0; i < N_SCAN; ++i)
        {
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }

    void resetVisualization(ros::Time this_stamp, std::string this_frame)
    {
        visualization_msgs::MarkerArray deleter;
        visualization_msgs::Marker deleter_marker;
        deleter_marker.header.frame_id = this_frame;
        deleter_marker.header.stamp = this_stamp;
        deleter_marker.action = visualization_msgs::Marker::DELETEALL;
        deleter.markers.push_back(deleter_marker);
        pub_meas_cluster_info.publish(deleter);
        pub_tracker_cluster_info.publish(deleter);

        jsk_recognition_msgs::BoundingBoxArray bbox_array;
        bbox_array.header.stamp = this_stamp;
        bbox_array.header.frame_id = this_frame;
        jsk_recognition_msgs::BoundingBox bbox;
        bbox.header.stamp = this_stamp;
        bbox.header.frame_id = this_frame;
        bbox_array.boxes.push_back(bbox);
        pub_bbox.publish(bbox_array);
    }

    void tracking()
    {
        static int lidar_count = -1;
        static int used_id_cnt = 0;
        if (++lidar_count % (0+1) != 0)
            return;
        
        meas_map.clear();
        resetVisualization(cloudHeader.stamp, "odom");

        // 1. Ground segmentation
        pcl::PointCloud<PointType>::Ptr nonground(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr ground(new pcl::PointCloud<PointType>());
        double patchwork_process_time;
        m_patchwork->estimate_ground(*extractedCloud, *ground, *nonground, patchwork_process_time);
        publishCloud(&pub_ground_cloud, ground, cloudHeader.stamp, "base_link");
        publishCloud(&pub_nonground_cloud, nonground, cloudHeader.stamp, "base_link");
        printf("[Query] Ground: %d, Nonground: %d, time: %f\n", (int)ground->points.size(), (int)nonground->points.size(), patchwork_process_time);
        
        static tf::TransformListener listener;
        static tf::StampedTransform init_transform;

        // 2. Find transform
        try{
            listener.waitForTransform("odom", "base_link", cloudHeader.stamp, ros::Duration(0.01));
            listener.lookupTransform("odom", "base_link", cloudHeader.stamp, init_transform);
        } 
        catch (tf::TransformException ex){
            ROS_ERROR("no imu tf");
            return;
        }

        // 3. Voxel and filter
        pcl::PointCloud<PointType>::Ptr cloud_filtered(new pcl::PointCloud<PointType>());
        TicToc voxel_time;
        static pcl::PassThrough<PointType> pass;
        pass.setInputCloud(nonground);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(minZ, maxZ);
        pass.filter(*cloud_filtered);
        printf("After z filter: %d\n", (int)cloud_filtered->points.size());
        *nonground = *cloud_filtered;

        // downsize_filter.setSaveLeafLayout(true);
        pcl::PointCloud<PointType>::Ptr nonground_ds(new pcl::PointCloud<PointType>());
        static pcl::VoxelGrid<PointType> downsize_filter;
        downsize_filter.setLeafSize(nongroundDownsample, nongroundDownsample, nongroundDownsample);
        downsize_filter.setInputCloud(nonground);
        downsize_filter.filter(*nonground_ds);
        ROS_WARN("Voxelize: %f ms", voxel_time.toc());
        printf("After voxelized (%f m): %d\n", nongroundDownsample, (int)nonground_ds->points.size());
        publishCloud(&pub_nonground_ds_cloud, nonground_ds, cloudHeader.stamp, "base_link");

        // 4. Get initial pose
        double xCur, yCur, zCur, rollCur, pitchCur, yawCur;
        xCur = init_transform.getOrigin().x();
        yCur = init_transform.getOrigin().y();
        zCur = init_transform.getOrigin().z();
        tf::Matrix3x3 init_m(init_transform.getRotation());
        init_m.getRPY(rollCur, pitchCur, yawCur);
        Eigen::Affine3f pose = pcl::getTransformation(xCur, yCur, zCur, rollCur, pitchCur, yawCur);
        pcl::transformPointCloud(*nonground_ds, *nonground_ds, pose);

        // 5. Clustering and fitting box
        meas_map.buildMap(nonground_ds, used_id_cnt);

        // 6. Init tracking
        TicToc tracking_time;
        std::vector<Cluster> meas_clusters = meas_map.getMap();
        int const n_targets = (int)meas_clusters.size();

        if (!tracker_flag)
        {
            gmphd::GaussianModel<4> birth;
            birth.m_weight = weightBirth; // 0.2
            birth.m_mean(0, 0) = meanXBirth; // 400.0
            birth.m_mean(1, 0) = meanYBirth; // 400.0
            birth.m_mean(2, 0) = 0.f;
            birth.m_mean(3, 0) = 0.f;
            birth.m_cov = MatrixXf::Identity(4, 4);
            birth.m_cov(0,0) *= (meanXBirth * meanXBirth);
            birth.m_cov(1,1) *= (meanXBirth * meanXBirth);
            birth.m_cov(2,2) *= (meanXBirth * meanXBirth);
            birth.m_cov(3,3) *= (meanXBirth * meanXBirth);
            gmphd::GaussianMixture<4> birth_model({birth});
            target_tracker.setBirthModel(birth_model);

            // Dynamics (motion model)
            target_tracker.setDynamicsModel(samplingPeriod, processNoise);

            // Detection model
            float const meas_background = 1 - detectionProb; // false detection prob
            target_tracker.setObservationModel(detectionProb, measPoseNoise, measSpeedNoise, meas_background);
        
            // Pruning parameters
            target_tracker.setPruningParameters(pruneThres, pruneMergeThres, maxGaussian);

            // Spawn
            gmphd::SpawningModel<4> spawn_model;
            std::vector<gmphd::SpawningModel<4>> spawns = {spawn_model};
            target_tracker.setSpawnModel(spawns);

            // Survival over time
            target_tracker.setSurvivalProbability(survivalProb); 

            tracker_flag = true;        
        }

        // 7. Set measurements and propagate
        std::vector<gmphd::Target<2>> target_meas;
        Eigen::Matrix<float, 2, 1> measurements;
        int detected = 0;
        for (int i = 0; i < n_targets; ++i)
        {
            if (meas_clusters[i].bbox.value >= 0)
            {
                measurements[0] = meas_clusters[i].bbox.pose.position.x;
                measurements[1] = meas_clusters[i].bbox.pose.position.y;
                target_meas.push_back({.position=measurements, .speed={0., 0.}, .weight=1., .id=meas_clusters[i].id});
                detected++;
            }
        }
    
        printf("Detections: %d out of %d\n", detected, (int)meas_clusters.size());
        target_tracker.setNewMeasurements(target_meas);
        target_tracker.propagate();

        // 8. Get tracker
        const auto tracked = target_tracker.getTrackedTargets(minWeightTrack);

        tracker_map.clear();
        std::vector<Cluster> track_clusters;
        for (size_t i = 0; i < tracked.size(); i++)
        {
            Cluster cluster;
            cluster.id = tracked[i].id;
            cluster.centroid_x = tracked[i].position[0];
            cluster.centroid_y = tracked[i].position[1];
            cluster.feature = sqrt(pow(tracked[i].speed[0], 2) + pow(tracked[i].speed[1], 2));
            cluster.vel_x = tracked[i].speed[0];
            cluster.vel_y = tracked[i].speed[1];
            track_clusters.push_back(cluster);
            printf("id: %d, tracked: %f;%f;%f, vel: %f;%f, weight: %f\n", tracked[i].id, tracked[i].position[0], tracked[i].position[1], 0.0, tracked[i].speed[0], tracked[i].speed[1], tracked[i].weight);
        }
        printf("meas size: %d, tracked size: %d\n", (int)target_meas.size(), (int)tracked.size());
        ROS_WARN("Tracking time: %f ms", tracking_time.toc());
        tracker_map.setMap(track_clusters);
        publishClusteredCloud(&pub_meas_cluster, &pub_meas_centroid, &pub_meas_cluster_info, meas_map, cloudHeader.stamp, "odom");
        publishClusteredCloud(&pub_tracker_cluster, &pub_tracker_centroid, &pub_tracker_cluster_info, tracker_map, cloudHeader.stamp, "odom");
        publishBoundingBox(&pub_bbox, meas_map, cloudHeader.stamp, "odom");
    }

    void publishClusteredCloud(ros::Publisher *thisPubCloud, ros::Publisher *thisPubCentroid, ros::Publisher *thisPubInfo, ClusterMap map, ros::Time thisStamp, string thisFrame)
    {
        pcl::PointCloud<PointType>::Ptr outPcl (new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr centroids (new pcl::PointCloud<PointType>);
        sensor_msgs::PointCloud2 segmentedCloud;
        sensor_msgs::PointCloud2 centroidCloud;

        visualization_msgs::MarkerArray ids;
        std::vector<Cluster> cluster_map = map.getMap();

        for (int i = 0; i < (int) cluster_map.size(); ++i)
        {
            Cluster cluster = cluster_map[i];
            *outPcl += cluster.cloud;
            PointType centroid;
            centroid.x = cluster.centroid_x;
            centroid.y = cluster.centroid_y;
            centroid.z = cluster.centroid_z;
            centroid.intensity = 100.0;
            centroids->points.emplace_back(centroid);
            
            if (map.getType() == TRACKER)
            {
                visualization_msgs::Marker velocity;
                velocity.header.frame_id = thisFrame;
                velocity.type = 0; // ARROWS
                geometry_msgs::Point pt1;
                pt1.x = centroid.x;
                pt1.y = centroid.y;
                geometry_msgs::Point pt2;
                pt2.x = pt1.x + cluster.vel_x * samplingPeriod;
                pt2.y = pt1.y + cluster.vel_y * samplingPeriod;
                velocity.scale.x = 0.1;
                velocity.scale.y = 0.1;
                velocity.color.a = 1.0;
                velocity.color.r = 1.0;
                velocity.action = 0;
                velocity.id = cluster_map.size() + i;
                velocity.points.push_back(pt1);
                velocity.points.push_back(pt2);
                ids.markers.push_back(velocity);
                
                visualization_msgs::Marker id;
                id.header.frame_id = thisFrame;
                id.scale.z = 0.8;
                id.color.r = 1.0;
                id.color.g = 0.0;
                id.color.b = 0.0;
                id.color.a = 1.0;
                id.action = 0;
                id.type = 9; // TEXT_VIEW_FACING
                id.id = cluster.id;
                id.text = to_string(cluster.id);
                id.pose.position.x = centroid.x;
                id.pose.position.y = centroid.y;
                id.pose.position.z = centroid.z;
                id.pose.orientation.w = 1.0;
                ids.markers.push_back(id);

                std::ostringstream stream;
                stream.precision(2);
                stream << "vx: " << cluster.vel_x << "\n"
                        << "vy: " << cluster.vel_y << "\n";
                std::string new_string = stream.str();

                visualization_msgs::Marker text;
                text.header.frame_id = thisFrame;
                text.scale.z = 0.3;
                text.color.r = 1.0;
                text.color.g = 1.0;
                text.color.b = 1.0;
                text.color.a = 1.0;
                text.action = 0;
                text.type = 9; // TEXT_VIEW_FACING
                text.id = 2*cluster_map.size() + i;
                text.text = new_string;
                text.pose.position.x = centroid.x - 0.3;
                text.pose.position.y = centroid.y - 0.3;
                text.pose.position.z = centroid.z;
                text.pose.orientation.w = 1.0;
                ids.markers.push_back(text);
            }
            else if (map.getType() == MEASUREMENT)
            {
                std::ostringstream stream;
                stream.precision(2);
                stream << "id: " << cluster.id << "\n"
                       << "h: " << cluster.m_height << "\n"
                       << "A: " << cluster.m_area << "\n"
                       << "R: " << cluster.m_ratio << "\n"
                       << "D: " << cluster.m_density<<"\n";
                std::string new_string = stream.str();

                visualization_msgs::Marker text;
                text.header.frame_id = thisFrame;
                text.scale.z = 0.3;
                text.color.r = 1.0;
                text.color.g = 1.0;
                text.color.b = 1.0;
                text.color.a = 1.0;
                text.action = 0;
                text.type = 9; // TEXT_VIEW_FACING
                text.id = 2*cluster_map.size() + i;
                text.text = new_string;
                text.pose.position.x = centroid.x - 0.8;
                text.pose.position.y = centroid.y - 0.8;
                text.pose.position.z = centroid.z;
                text.pose.orientation.w = 1.0;
                ids.markers.push_back(text);
            }
        }
        pcl::toROSMsg(*outPcl, segmentedCloud);
        pcl::toROSMsg(*centroids, centroidCloud);
        segmentedCloud.header.stamp = thisStamp;
        segmentedCloud.header.frame_id = thisFrame;
        centroidCloud.header = segmentedCloud.header;

        if (thisPubInfo->getNumSubscribers() != 0)
            thisPubInfo->publish(ids);
        if (thisPubCloud->getNumSubscribers() != 0)
            thisPubCloud->publish(segmentedCloud);
        if (thisPubCentroid->getNumSubscribers() != 0)
            thisPubCentroid->publish(centroidCloud);
    }

    void publishBoundingBox(ros::Publisher *thisPub, ClusterMap map, ros::Time timestamp, string this_frame)
    {
        if (thisPub->getNumSubscribers() != 0)
        {
            jsk_recognition_msgs::BoundingBoxArray bbox_array;
            bbox_array.header.stamp = timestamp;
            bbox_array.header.frame_id = this_frame;
            std::vector<Cluster> cluster_map = map.getMap();
            for (size_t i = 0; i < cluster_map.size(); ++i)
            {
                jsk_recognition_msgs::BoundingBox bbox = cluster_map[i].bbox;
                // if (bbox.value >= 0)
                // {
                    bbox.header.stamp = timestamp;
                    bbox.header.frame_id = this_frame;
                    bbox_array.boxes.push_back(bbox);  
                // }
            }
            pub_bbox.publish(bbox_array);
        }
    }

    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed  = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    ImageProjection IP;
    
    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    
    return 0;
}
