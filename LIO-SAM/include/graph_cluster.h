#pragma once
#ifndef _GRAPH_CLUSTER_H_
#define _GRAPH_CLUSTER_H_

#include "utility.h"

using namespace std;
using namespace Eigen;

class Node
{
    public:
        uint channel;
        uint start;
        uint end;
        int label;
        vector<Node> adjacent;

        Node() : label(-1){}
};

class Point
{
    public:
        uint idx;
        float x, y, z, intensity, range;
        float vert, hori;
        bool valid;
        int first_label;
        int final_label;

        Point() : valid(false), first_label(-1), final_label(-1) {}
};

class GraphCluster
{
    public: 
        int vert_scan;
        int horz_scan;
        float min_range;
        float max_range;
        float dist_thres;
        float angle_thres;
        int downsample;
        vector<float> vert_angles;
        vector<vector<Point>> range_mat;
        vector<Node> graph;

    GraphCluster(int _vert_scan, int _horz_scan, float _min_range, float _max_range, int _downsample, vector<float> _vert_angles,
                float _dist_thres, float _angle_thres) {
        vert_scan = _vert_scan;
        horz_scan = _horz_scan;
        min_range = _min_range;
        max_range = _max_range;
        vert_angles = _vert_angles;
        downsample = _downsample;
        dist_thres = _dist_thres;
        angle_thres = _angle_thres;
        range_mat.resize(vert_scan, std::vector<Point>(horz_scan));
    }

    int getRowIndex(float x, float y, float z) {
        float vert_angle = atan2(z, sqrt(x*x + y*y))*180/M_PI;
        auto iter_geq = std::lower_bound(vert_angles.begin(), vert_angles.end(), vert_angle);
        int row_idx;

        if (iter_geq == vert_angles.begin())
        {
            row_idx = 0;
        }
        else
        {
            float a = *(iter_geq - 1);
            float b = *(iter_geq);
            if (fabs(vert_angle-a) < fabs(vert_angle-b)){
                row_idx = iter_geq - vert_angles.begin() - 1;
            } else {
                row_idx = iter_geq - vert_angles.begin();
            }
        }
        return row_idx;
    }

    int getColIndex(float x, float y) {
        float horizonAngle = atan2(x, y) * 180 / M_PI;

        static float ang_res_x = 360.0/float(horz_scan);

        int col_idx = -round((horizonAngle-90.0)/ang_res_x) + horz_scan/2;
        if (col_idx >= horz_scan)
            col_idx -= horz_scan;
        return col_idx;
    }

    void setInputCloud(pcl::PointCloud<PointType>::Ptr cloud_in) {
        
        for (size_t i = 0; i < cloud_in->points.size(); i++) {
            Point point;
            point.x = cloud_in->points[i].x;
            point.y = cloud_in->points[i].y;
            point.z = cloud_in->points[i].z;
            point.idx = i;
            point.intensity = cloud_in->points[i].intensity;
            point.range = sqrt(point.x*point.x + point.y*point.y + point.z*point.z);
            point.valid = true;
            
            if (point.range < min_range || point.range > max_range)
                continue;

            int row_idx = getRowIndex(point.x, point.y, point.z);
            
            if (row_idx < 0 || row_idx >= vert_scan)
                continue;

            int col_idx = getColIndex(point.x, point.y);

            if (col_idx < 0 || col_idx >= horz_scan)
                continue;

            if (range_mat[row_idx][col_idx].valid)
                continue;

            range_mat[row_idx][col_idx] = point;
        }
    }

    float calcDistance(Point pt1, Point pt2) {
        return sqrt((pt1.x - pt2.x)*(pt1.x-pt2.x) + (pt1.y-pt2.y)*(pt1.y-pt2.y));
    }

    bool calcAngles(Point pt1, Point adj1, Point pt2, Point adj2) {
        Vector2f v1(adj1.x - pt1.x, adj1.y - pt1.y);
        Vector2f v2(adj2.x - pt2.x, adj2.y - pt2.y);
        Vector2f v_mid(-(pt1.x+pt2.x)/2.0, -(pt1.y+pt2.y)/2.0);
        v1.normalize();
        v2.normalize();
        Vector2f v_bisector = (v1 + v2).normalized();
        bool is_valid = (v_mid.dot(v_bisector) > 0);
        float angle = acos(v1.dot(v2)) * 180.0 / M_PI;
        is_valid = is_valid && (angle < angle_thres);
        return is_valid;
    }

    void extract() {
        int count = 0;
        for (int i = 0; i < vert_scan; i++) {
            if (i % downsample != 0)
                continue;
            int start_pos = -1;
            int pointer = -1;
            int end_pos = -1;
            
            for(int j = 0; j < horz_scan; j++) {
                
                Node prev_node;

                if (!range_mat[i][j].valid) continue;
                
                if (count == 0) {
                    count++;
                    start_pos = j;
                    pointer = j;
                    range_mat[i][j].first_label = count;
                } else {
                    float dist = calcDistance(range_mat[i][j], range_mat[i][pointer]);

                    if (dist < dist_thres) {
                        // find adjacent points
                        Point adj1;
                        Point adj2;
                        
                        if (pointer == 0) { 
                            adj1 = range_mat[i][horz_scan-1];
                        } else { 
                            adj1 = range_mat[i][pointer-1];
                        }    
                        
                        if (j == horz_scan-1) { 
                            adj2 = range_mat[i][0]; 
                        } else { 
                            adj2 = range_mat[i][j+1];
                        }

                        // Same cluster if either adjcents is not valid
                        if (!adj1.valid || !adj1.valid) {
                            pointer = j;
                            range_mat[i][j].first_label = count;
                            continue;    
                        } 

                        // different cluster despite close distance
                        if (calcAngles(range_mat[i][pointer], adj1, range_mat[i][j], adj2)) {
                            
                            // push to graph
                            Node node;
                            node.channel = i;
                            node.start = start_pos;
                            node.end = end_pos;
                            node.label = count;
                            if (prev_node.label < 0) {
                                prev_node = node;
                            } else {
                                node.adjacent.push_back(prev_node);
                                prev_node.adjacent.push_back(node);
                                graph.push_back(prev_node);
                            }

                            end_pos = pointer;
                            count++;
                            start_pos = j;
                            pointer = j;
                            range_mat[i][j].first_label = count;

                        } else { // same cluster
                            pointer = j; 
                            range_mat[i][j].first_label = count;
                        }
                    } else { // differrent cluster
                        end_pos = pointer;
                        count++;
                        start_pos = j;
                        pointer = j;
                        range_mat[i][j].first_label = count;
                    }
                }
            }
            ROS_WARN("Horizontal count: %d", count);
        }
    }
};


#endif 