#pragma once
#ifndef _GRAPH_CLUSTER_H_
#define _GRAPH_CLUSTER_H_

#include "utility.h"
#include <forward_list>

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
        int label;
        int first_label;
        int final_label;

        Point() : valid(false), first_label(-1), final_label(-1), label(0) {}
};


class GraphCluster : public ParamServer
{
    private:
        uint16_t max_label_;
        vector<forward_list<Point*>> runs_;
        forward_list<Point*> dummy_;
        vector<vector<int>> valid_idx_;
        vector<int> valid_cnt_;
        vector<vector<Node>> nodes_;

    public: 
        vector<float> vert_angles;
        vector<vector<Point>> range_mat;
        vector<Node> graph;

        ros::Publisher node_pub;
        ros::Publisher cloud_pub;

    GraphCluster(vector<float> _vert_angles) {
        vert_angles = _vert_angles;
        range_mat.resize(N_SCAN, vector<Point>(Horizon_SCAN));
        valid_idx_.resize(N_SCAN, vector<int>());
        nodes_.resize(N_SCAN, vector<Node>());
        valid_cnt_.resize(N_SCAN, 0);
        dummy_ = {};
        
        node_pub = nh.advertise<visualization_msgs::MarkerArray>("/tracking/lidar/nodes", 1);
        cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/tracking/lidar/labelled", 1);
    }

    void publishNodes() {
        
        visualization_msgs::MarkerArray markers;
        visualization_msgs::MarkerArray deleter;
        visualization_msgs::Marker deleter_marker;
        deleter_marker.header.frame_id = lidarFrame;
        deleter_marker.header.stamp = ros::Time::now();
        deleter_marker.action = visualization_msgs::Marker::DELETEALL;
        deleter.markers.push_back(deleter_marker);
        node_pub.publish(deleter);
 
        int cnt = 1000;
        for (size_t i = 0; i < nodes_.size(); i++) {
            if (nodes_[i].size() == 0)
                continue;
            
            for (size_t j = 0; j < nodes_[i].size(); j++){
                visualization_msgs::Marker text;
                std::ostringstream stream;
                stream.precision(2);
                stream << i <<" " << nodes_[i][j].start << " " << nodes_[i][j].end << endl;
                std::string new_string = stream.str();

                text.header.frame_id = lidarFrame;
                text.scale.z = 0.2;
                text.color.r = 1.0;
                text.color.g = 1.0;
                text.color.b = 1.0;
                text.color.a = 1.0;
                text.action = 0;
                text.type = 9; // TEXT_VIEW_FACING
                text.id = cnt;
                text.text = new_string;
                text.pose.position.x = range_mat[i][nodes_[i][j].start].x;
                text.pose.position.y = range_mat[i][nodes_[i][j].start].y;
                text.pose.position.z = range_mat[i][nodes_[i][j].start].z;
                text.pose.orientation.w = 1.0;
                markers.markers.push_back(text);
                cnt++;
            }
        }
        node_pub.publish(markers);
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

        static float ang_res_x = 360.0/float(Horizon_SCAN);

        int col_idx = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
        if (col_idx >= Horizon_SCAN)
            col_idx -= Horizon_SCAN;
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
            
            if (point.range < lidarMinRange || point.range > lidarMaxRange)
                continue;

            int row_idx = getRowIndex(point.x, point.y, point.z);
            
            if (row_idx < 0 || row_idx >= N_SCAN)
                continue;

            int col_idx = getColIndex(point.x, point.y);

            if (col_idx < 0 || col_idx >= Horizon_SCAN)
                continue;

            if (range_mat[row_idx][col_idx].valid) {
                continue;
            }

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
        is_valid = is_valid && (angle < graphAngleThres);
        return is_valid;
    }

    void setInputCloud2(pcl::PointCloud<PointType>::Ptr cloud_in, vector<vector<int>> &clusters) {

        max_label_ = 1;
        runs_.clear();
        runs_.push_back(dummy_); // dummy for index `0`
        runs_.push_back(dummy_);

        for (int i = 0; i < N_SCAN; i++) {
            valid_idx_[i].clear();
        }

        std::for_each(range_mat.begin(), range_mat.end(), [](vector<Point>& inner_vec) {
            std::fill(inner_vec.begin(), inner_vec.end(), Point());
        });


        TicToc organize_time;
        // organize points
        for (size_t i = 0; i < cloud_in->points.size(); i++) {
            auto &pt = cloud_in->points[i];
            Point point;
            point.x = pt.x;
            point.y = pt.y;
            point.z = pt.z;
            point.idx = i;
            point.range = sqrt(point.x*point.x + point.y*point.y + point.z*point.z);
            point.valid = true;

            if (point.range < lidarMinRange || point.range > lidarMaxRange) {
                pt.intensity = -1;
                continue;
            }

            int row_idx = getRowIndex(point.x, point.y, point.z);
            
            if (row_idx < 0 || row_idx >= N_SCAN) {
                pt.intensity = -1;
                continue;
            }

            int col_idx = getColIndex(point.x, point.y);

            if (col_idx < 0 || col_idx >= Horizon_SCAN){
                pt.intensity = -1;
                continue;
            }

            if (range_mat[row_idx][col_idx].valid) {
                pt.intensity = -1;
                continue;
            }
            range_mat[row_idx][col_idx] = point;
            valid_idx_[row_idx].push_back(col_idx);
            valid_cnt_[row_idx]++;
            // sort(valid_idx_[row_idx].begin(), valid_idx_[row_idx].end());
        }
        ROS_WARN("Organize: %f ms", organize_time.toc());

        TicToc process_time;
        // main processing
        for (int i = 0; i < N_SCAN; i++) {

            // if (i % downsampleRate != 0)
            //     continue;

            TicToc each_run_time;
            // find_runs(i);
            find_runs2(i);
            // ROS_WARN("find run: %f ms", each_run_time.toc());

            // if (cluster_debug) {
            //     for (int i = 0; i < runs_.size(); ++i) {
            //         for (auto &p : runs_[i]) {
            //             cloud_in->points[p->idx].intensity = i;
            //             printf("F label: %d -- %f;%f;%f\n", i, p->x, p->y, p->z);
            //         }
            //     }
            //     publishCloud(&cloud_pub, cloud_in, ros::Time::now(), lidarFrame);
            //     publishNodes();
            //     cin.get();
            // }

            TicToc update_run_time;
            update_labels2(i);
            // ROS_WARN("update run: %f ms", update_run_time.toc());
            // if (cluster_debug) {
            //     for (int i = 0; i < runs_.size(); ++i) {
            //         for (auto &p : runs_[i]) {
            //             cloud_in->points[p->idx].intensity = i;
            //             printf("U label: %d -- %f;%f;%f\n", i, p->x, p->y, p->z);
            //         }
            //     }
            //     publishCloud(&cloud_pub, cloud_in, ros::Time::now(), lidarFrame);
            //     cin.get();
            // }
        }
        ROS_WARN("Process: %f ms", process_time.toc());

        TicToc conversion_time;
        // convert runs to indices
        if (cluster_debug) 
            printf("----------Result: %d------------------------------\n", runs_.size());
        
        int cnt = 0;
        for (int i = 0; i < runs_.size(); ++i) {
            for (auto &p : runs_[i]) {
                cloud_in->points[p->idx].intensity = i;
                if (cluster_debug) {
                    printf("Label: %d -- %f;%f;%f\n", i, p->x, p->y, p->z);
                }
            }

            if (std::distance(runs_[i].begin(), runs_[i].end()) > 0)
                cnt++;
        }

        ROS_WARN("# of cluster: %d", cnt);
        ROS_WARN("Conversion time: %f ms", conversion_time.toc());
    }

    void find_runs(int scan_line) {
        size_t point_size = valid_idx_[scan_line].size();
        if (point_size <= 0) 
            return;

        int first_valid_idx = valid_idx_[scan_line][0];
        int last_valid_idx = valid_idx_[scan_line][point_size - 1];
        printf("scan line: %d\n", scan_line);
        for (int j = 0; j < point_size-1; j++) {
            
            int c_idx = valid_idx_[scan_line][j];
            int n_idx = valid_idx_[scan_line][j+1];

            if (j == 0) {
                // first point
                auto &p_0 = range_mat[scan_line][c_idx];
                max_label_ += 1;
                runs_.push_back(dummy_);
                range_mat[scan_line][c_idx].label = max_label_;
                runs_[p_0.label].insert_after(runs_[p_0.label].cbefore_begin(), &range_mat[scan_line][c_idx]);
                if (p_0.label == 0)
                    ROS_ERROR("p_0.label == 0");
            }
            
            // compare with the next point
            auto &p_c = range_mat[scan_line][c_idx];
            auto &p_n = range_mat[scan_line][n_idx];
            printf("p_c: %f;%f;%f p_n: %f;%f;%f\n", p_c.x, p_c.y, p_c.z, p_n.x, p_n.y, p_n.z);
            // Same run within same distance threshold
            // else make a new run
            if (calcDistance(p_c, p_n) < graphDistThres) {
                p_n.label = p_c.label;
                printf("merged\n");
            } else {
                max_label_++;
                p_n.label = max_label_;
                runs_.push_back(dummy_);
            }
            // insert next point
            runs_[p_n.label].insert_after(runs_[p_n.label].cbefore_begin(), &range_mat[scan_line][n_idx]);
            if (p_n.label == 0)
                ROS_ERROR("p_n.label == 0");
        } 

        // compare last and first point
        if (point_size > 2) {
            auto &p_0 = range_mat[scan_line][first_valid_idx];
            auto &p_l = range_mat[scan_line][last_valid_idx];
            if (calcDistance(p_0, p_l) < graphDistThres) {
                if (p_0.label == 0) 
                    ROS_ERROR("Ring merge to label 0");
                if (p_0.label != p_l.label) {
                    merge_runs(p_l.label, p_0.label);
                }
            }
        } else if (point_size == 1) {
            // Only one point -> make a new run
            auto &p_0 = range_mat[scan_line][first_valid_idx];
            max_label_ += 1;
            runs_.push_back(dummy_);
            range_mat[scan_line][first_valid_idx].label = max_label_;
            runs_[p_0.label].insert_after(runs_[p_0.label].cbefore_begin(), &range_mat[scan_line][first_valid_idx]);
        }
    }

    void update_labels(int scan_line) {
        // Iterate each point of this scan line to update the labels.
        int point_size_j_idx = valid_idx_[scan_line].size();
        // Current scan line is emtpy, do nothing.
        if(point_size_j_idx==0) return;
        
        // Iterate each point of this scan line to update the labels.
        for(int j_idx=0;j_idx<point_size_j_idx;j_idx++){
            
            int j = valid_idx_[scan_line][j_idx];

            auto &p_j = range_mat[scan_line][j];

            for(int l=scan_line-1;l>=0;l--){
                if(valid_idx_[l].size()==0)
                    continue;

                // Smart index for the near enough point, after re-organized these points.
                int nn_idx = j;

                if(!range_mat[l][nn_idx].valid)
                    continue;

                // Nearest neighbour point
                auto &p_nn = range_mat[l][nn_idx];
                // printf("nearest_idx: %d, %f;%f;%f\n", nn_idx, p_nn.x, p_nn.y, p_nn.z);

                // Skip, if these two points already belong to the same run.
                if(p_j.label == p_nn.label){
                    continue;
                }
                double dist_min = calcDistance(p_j, p_nn);

                /* Otherwise,
                If the distance of the `nearest point` is within `th_merge_`, 
                then merge to the smaller run.
                */
                if(dist_min < graphMergeThres){
                    uint16_t  cur_label = 0, target_label = 0;

                    if(p_j.label ==0 || p_nn.label==0){
                        ROS_ERROR("p_j.label:%u, p_nn.label:%u", p_j.label, p_nn.label);
                    }
                    // Merge to a smaller label cluster
                    if(p_j.label > p_nn.label){
                        cur_label = p_j.label;
                        target_label = p_nn.label;
                    }else{
                        cur_label = p_nn.label;
                        target_label = p_j.label;
                    }

                    // Merge these two runs.
                    merge_runs(cur_label, target_label);
                }
            }
        }
    }

    void merge_runs(uint16_t cur_label, uint16_t target_label) {
        if(cur_label == 0 || target_label == 0){
            printf("Error merging runs cur_label:%u target_label:%u", cur_label, target_label);
            ROS_ERROR("Error merging runs cur_label:%u target_label:%u", cur_label, target_label);
        }
        // printf("run total size: %d\n", runs_.size());
        // First, modify the label of current run.
        // printf("run size at cur_label: %d\n", std::distance(runs_[cur_label].begin(), runs_[cur_label].end()));
        for(auto &p : runs_[cur_label]){
            p->label = target_label;
        }
        // Then, insert points of current run into target run.
        runs_[target_label].insert_after(runs_[target_label].cbefore_begin(), runs_[cur_label].begin(),runs_[cur_label].end() );
        runs_[cur_label].clear();
        // printf("run size at target_label: %d\n", std::distance(runs_[target_label].begin(), runs_[target_label].end()));
        // printf("run size at cur_label: %d\n", std::distance(runs_[cur_label].begin(), runs_[cur_label].end()));        
    }

    void find_runs2(int scan_line) {

        nodes_[scan_line].clear();

        if (valid_cnt_[scan_line] <= 0)
            return;

        int first_valid_idx = -1;
        int last_valid_idx = -1;
        int start_pos = -1;
        int pre_pos = -1;
        int end_pos = -1;
        
        Node node;
        bool first_node = false;
        for(int j = 0; j < Horizon_SCAN; j++) {

            if (!range_mat[scan_line][j].valid) continue;

            if (!first_node) {
                first_node = true;
                // update index
                first_valid_idx = j;
                start_pos = j;
                pre_pos = j;
                // update label
                max_label_++;
                runs_.push_back(dummy_);
                range_mat[scan_line][j].label = max_label_;

                // push a new node
                node.start = start_pos;
                node.end = j;
                node.label = range_mat[scan_line][j].label;
                nodes_[scan_line].push_back(node);
                runs_[node.label].insert_after(runs_[node.label].cbefore_begin(), &range_mat[scan_line][j]);
            } else {
                auto &cur_pt = range_mat[scan_line][j];
                auto &pre_pt = range_mat[scan_line][pre_pos]; 
                if (calcDistance(cur_pt, pre_pt) < graphDistThres) {
                    // update existing node
                    pre_pos = j; 
                    cur_pt.label = pre_pt.label;
                    nodes_[scan_line].back().end = j;
                } else { 
                    // update index
                    start_pos = j;
                    pre_pos = j;
                    // update label
                    max_label_++;
                    runs_.push_back(dummy_);
                    cur_pt.label = max_label_;
                    // push new node
                    node.start = start_pos;
                    node.end = j;
                    node.label = range_mat[scan_line][j].label;
                    nodes_[scan_line].push_back(node);
                    assert(range_mat[scan_line][j].label == cur_pt.label);
                }
                runs_[cur_pt.label].insert_after(runs_[cur_pt.label].cbefore_begin(), &range_mat[scan_line][j]);
            }
        }
        last_valid_idx = pre_pos;

        // merge last and first
        if (nodes_[scan_line].size() > 2) {
            auto &p_0 = range_mat[scan_line][first_valid_idx];
            auto &p_l = range_mat[scan_line][last_valid_idx];
            if (calcDistance(p_0, p_l) < graphDistThres) {
                if (p_0.label == 0) 
                    ROS_ERROR("Ring merge to label 0");
                if (p_0.label != p_l.label) {
                    nodes_[scan_line].back().label = p_0.label;
                    merge_runs(p_l.label, p_0.label);
                }
            }
        }

        // check if merge between nodes 
        if (cluster_debug) {
            printf("-------------------Channel: %d------------------------------\n", scan_line);
            printf("first: %d, last: %d\n", first_valid_idx, last_valid_idx);
            for (auto n : nodes_[scan_line]){
                printf("label: %d, start: %d, end: %d\n", n.label, n.start, n.end);  
                for (auto el : runs_[n.label]) {
                    printf("%f;%f;%f\n", el->x, el->y, el->z);
                } 
            }
        }

        // merge skipped nodes due to occlusion
        if (nodes_[scan_line].size() > 2) {
            uint16_t cur_label = 0, target_label = 0;
            for (size_t i = 0; i < nodes_[scan_line].size()-1; i++){
                for (size_t j = i+1; j < nodes_[scan_line].size(); j++) {
                    auto &node_i = nodes_[scan_line][i];
                    auto &node_j = nodes_[scan_line][j];
                    
                    if (node_i.label == node_j.label)
                        continue;

                    int end_idx = node_i.end;
                    int start_idx = node_j.start;
                    if (calcDistance(range_mat[scan_line][end_idx], range_mat[scan_line][start_idx]) < graphDistThres) {
                        if (node_i.label > node_j.label) {
                            target_label = node_j.label;
                            cur_label = node_i.label;
                            node_i.label = target_label;
                        } else {
                            target_label = node_i.label;
                            cur_label = node_j.label;
                            node_j.label = target_label;
                        }
                        if (cluster_debug)
                            printf("merge two labels despite occlusion: %d %d\n", cur_label, target_label);
                        merge_runs(cur_label, target_label);
                        // merge !
                    }
                }
            }
        }   
    }

    bool mergeNodes(Node &first_node, Node &second_node, int cur_line, int prev_line, int query_idx) {
        if (query_idx >= first_node.start && query_idx <= first_node.end) {
            if (range_mat[cur_line][query_idx].valid) {
                int left_idx = query_idx;
                int right_idx = query_idx;
                while (1) {
                    if (left_idx <= query_idx - searchWindowSize && right_idx >= query_idx + searchWindowSize)
                        break;

                    if (left_idx >= second_node.start && range_mat[prev_line][left_idx].valid) {
                        if (checkDistance(first_node, second_node, range_mat[cur_line][query_idx], range_mat[prev_line][left_idx]))
                            return true;
                    } 

                    if (right_idx <= second_node.end && range_mat[prev_line][right_idx].valid) {
                        if (checkDistance(first_node, second_node, range_mat[cur_line][query_idx], range_mat[prev_line][right_idx]))
                            return true;
                    }
                    left_idx--;
                    right_idx++;
                }
            }
        }
        return false;
    }

    bool checkDistance(Node &first_node, Node &second_node, Point &first_point, Point &second_point) {
        uint16_t cur_label, target_label = 0;

        if (first_node.label == second_node.label)
            return false;

        if (calcDistance(first_point, second_point) < graphMergeThres) {
            if (cluster_debug) {
                printf("Two nodes merged: %d, %d\n", first_node.label, second_node.label);
                // printf("Two labels merged: %d, %d\n", first_point.label, second_point.label);
            }

            if (first_node.label > second_node.label) {
                cur_label = first_node.label;
                target_label = second_node.label;
                first_node.label = target_label;
            } else {
                cur_label = second_node.label;
                target_label = first_node.label;
                second_node.label = target_label;
            }
            merge_runs(cur_label, target_label);
            return true;
        }
        return false;
    }

    bool overlap(Node &first_node, Node &second_node) {
        int new_start = first_node.start - searchWindowSize;
        int new_end = first_node.end + searchWindowSize;
        if (new_start <= second_node.start && second_node.start <= new_end) 
            return true;
        else if (new_start <= second_node.end && second_node.end <= new_end)
            return true;
        else if (second_node.start <= new_start && new_end <= second_node.end)
            return true;
        return false;
    }

    void update_labels2(int scan_line) {

        // Iterate each point of this scan line to update the labels.
        int point_size = valid_cnt_[scan_line];
        // Current scan line is emtpy, do nothing.
        if(point_size == 0) return;
        // Skip first scan line
        if(scan_line == 0) return;

        int prev_line = scan_line - 1;

        for(int n = 0; n < nodes_[scan_line].size(); n++) {
            
            auto &cur_node = nodes_[scan_line][n];

            for (int l = prev_line; l >= scan_line - graphLookupSize*1; l -= 1) {
                
                if (l < 0) // out of range
                    break;

                if (cluster_debug) {
                    printf("---cur line: %d, prev_line: %d\n", scan_line, l);
                    printf("cur_node start, end: %d, %d\n", cur_node.start, cur_node.end);
                }

                if (valid_cnt_[l] == 0)
                    continue;

                // binary search lower bound
                // lower_bnd inclusive
                int N = nodes_[l].size();
                int first = 0;
                int last = N - 1;
                int lower_bnd = 0;
                int mid = 0;
                
                while (first <= last) {
                    mid = (first + last) / 2;
                    auto &prev_node = nodes_[l][mid];
                    if (overlap(cur_node, prev_node) || cur_node.end < prev_node.start){
                        lower_bnd = mid;
                        last = mid - 1;
                    } else {
                        first = mid + 1;
                    }
                } 

                // binary search upper bound
                // exclusive but gives -1 if end of list
                first = 0;
                last = N - 1;
                mid = 0;
                int upper_bnd = 0;

                while (first <= last) {
                    mid = (first + last) / 2;
                    auto &prev_node = nodes_[l][mid];
                    if (overlap(cur_node, prev_node) || prev_node.end < cur_node.start) {
                        upper_bnd = mid;
                        first = mid + 1;
                    } else {
                        last = mid - 1;
                    }
                }
                upper_bnd = upper_bnd + 1;

                if (cluster_debug) {
                    printf("lower bound: %d, start, end: %d, %d\n", lower_bnd, nodes_[l][lower_bnd].start, nodes_[l][lower_bnd].end);
                    if (upper_bnd >= N)
                        printf("upper bound: out of range\n");
                    else
                        printf("upper bound: %d, start, end: %d, %d\n", upper_bnd, nodes_[l][upper_bnd-1].start, nodes_[l][upper_bnd-1].end);
                }

                // loop through overlapped nodes
                for (size_t idx = lower_bnd; idx < upper_bnd; idx++) {

                    auto &ovl_node = nodes_[l][idx];

                    if (ovl_node.label == cur_node.label)
                        continue;

                    if (cluster_debug)
                        printf("overlapped: %d, %d\n", ovl_node.start, ovl_node.end);

                    int iter_start_idx = -1;
                    int iter_end_idx = -1;

                    if (ovl_node.start <= cur_node.start && cur_node.end <= ovl_node.end) { 
                        // cur_node inside prev_node
                        iter_start_idx = cur_node.start;
                        iter_end_idx = cur_node.end;
                    } else if (cur_node.start <= ovl_node.start && ovl_node.end <= cur_node.end) {
                        // prev_node inside cur_node 
                        iter_start_idx = ovl_node.start;
                        iter_end_idx = ovl_node.end;
                    } else if (cur_node.start < ovl_node.start && cur_node.end >= ovl_node.start && cur_node.end <= ovl_node.end) {
                        // tail of cur_node with head of prev_node 
                        iter_start_idx = ovl_node.start;
                        iter_end_idx = cur_node.end;
                    } else if (ovl_node.start <= cur_node.start && cur_node.start <= ovl_node.end && cur_node.end > ovl_node.end) {
                        // head of cur_node with tail of prev_node
                        iter_start_idx = cur_node.start;
                        iter_end_idx = ovl_node.end;
                    } else {
                        // overlapped within search window size, use euclidean distance directly
                        if (ovl_node.end < cur_node.start) {
                            checkDistance(cur_node, ovl_node, range_mat[scan_line][cur_node.start], range_mat[l][ovl_node.end]);
                        } else if (cur_node.end < ovl_node.start) {
                            checkDistance(cur_node, ovl_node, range_mat[scan_line][cur_node.end], range_mat[l][ovl_node.start]);
                        }
                        if (cluster_debug)
                            printf("Merge by euclidean distance\n");
                        
                        continue;
                    }
                    
                    if (cluster_debug)
                        printf("overlapping: %d %d\n", iter_start_idx, iter_end_idx);

                    // iterate through overlapping indices
                    uint16_t cur_label = 0, target_label = 0;
                    
                    bool merged = false;

                    // Option #1
                    int cur_start_left = iter_start_idx;
                    int cur_start_right = iter_start_idx;
                    int cur_end_left = iter_end_idx;
                    int cur_end_right = iter_end_idx;

                    while (1) {
                        
                        if (cur_start_right > cur_end_left && cur_start_left < iter_start_idx - searchWindowSize && cur_end_right > iter_end_idx + searchWindowSize) // end of search
                            break;

                        if (mergeNodes(cur_node, ovl_node, scan_line, l, cur_start_left))
                            break;
                        
                        if (cur_start_left != cur_end_left) { // more than one overlapping cur_node point 
                            if (mergeNodes(cur_node, ovl_node, scan_line, l, cur_end_left))
                                break;
                        }


                        if (cur_start_left != cur_start_right) { // not the first iteration
                            if (mergeNodes(cur_node, ovl_node, scan_line, l, cur_start_right))
                                break;
                        }

                        if (cur_end_left != cur_end_right) { // not the first iteration
                            if (mergeNodes(cur_node, ovl_node, scan_line, l, cur_end_right))
                                break;
                        }

                        cur_start_left--;
                        cur_start_right++;
                        cur_end_left--;
                        cur_end_right++;
                    }
                }
            }
        }
    }

    void extract() {
        int count = 0;
        for (int i = 0; i < N_SCAN; i++) {
            if (i % downsampleRate != 0)
                continue;
            int start_pos = -1;
            int pointer = -1;
            int end_pos = -1;
            
            for(int j = 0; j < Horizon_SCAN; j++) {
                
                Node prev_node;

                if (!range_mat[i][j].valid) continue;
                
                if (count == 0) {
                    count++;
                    start_pos = j;
                    pointer = j;
                    range_mat[i][j].first_label = count;
                } else {
                    float dist = calcDistance(range_mat[i][j], range_mat[i][pointer]);

                    if (dist < graphDistThres) {
                        // find adjacent points
                        Point adj1;
                        Point adj2;
                        
                        if (pointer == 0) { 
                            adj1 = range_mat[i][Horizon_SCAN-1];
                        } else { 
                            adj1 = range_mat[i][pointer-1];
                        }    
                        
                        if (j == Horizon_SCAN-1) { 
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
                        // node 추가
                    }
                }
            }
            ROS_WARN("Horizontal count: %d", count);
        }
    }
};


#endif 