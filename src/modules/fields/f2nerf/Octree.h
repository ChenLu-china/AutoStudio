/**
* This file is part of autostudio
* Copyright (C) 
**/


#ifndef OCTREE_H
#define OCTREE_H
#include <torch/torch.h>
#include "Eigen/Eigen"
#include "../../../Common.h"
#include "../../../dataset/Dataset.h"

namespace AutoStudio
{

using Tensor = torch::Tensor;
#define INIT_NODE_STAT 1000
#define N_PROS 12
#define PersMatType Eigen::Matrix<float, 2, 4, Eigen::RowMajor>
#define TransWetType Eigen::Matrix<float, 3, N_PROS, Eigen::RowMajor>

struct alignas(32) OctreeTransInfo {
  PersMatType w2xz[N_PROS];
  TransWetType weight;
  Wec3f center;
  float dis_summary;
};


struct alignas(32) OctreeNode
{
    Wec3f center_;
    float extend_len_;
    int parent_;
    int child_[8];
    bool is_leaf_node_;
    int trans_idx_;
};

struct alignas(32) OctreeEdge
{
    int t_idx_a_;
    int t_idx_b_;
    Wec3f center_;
    Wec3f dir_0_;
    Wec3f dir_1_;
};

class Octree
{
private:
    /* data */
public:
    Octree(int max_depth, float bbox_side_len, float split_dist_thres, Dataset* dataset);
    
    float DistanceSummary(const Tensor& dis);
    std::vector<int> GetVaildCams(float bbox_len, const Tensor& center);

    inline void GenPixelIdx();
    inline void AddTreeEdges();
    inline void AddTreeNode(int u, int depth, Wec3f center, float bbox_len);
    inline OctreeTransInfo AddTreeTrans(const Tensor& rand_pts,
                                        const Tensor& c2w, 
                                        const Tensor& intri, 
                                        const Tensor& center);
    void ProcOctree(bool compact, bool subdivide, bool brute_force);
    void MarkInvisibleNodes();
    
public:   
    int max_depth_;
    Tensor c2w_, w2c_, intri_, bound_;
    Tensor cam_coords_;
    float bbox_len_;
    float dist_thres_;

    std::vector<OctreeNode> octree_nodes_;
    Tensor octree_nodes_gpu_;
    
    std::vector<OctreeTransInfo> octree_trans_; 
    Tensor octree_trans_gpu_;
    
    std::vector<OctreeEdge> octree_edges_;
    Tensor octree_edges_gpu_;

    Tensor tree_weight_stats_, tree_visit_cnt_;
    Tensor tree_alpha_stats_, node_search_order_;
    Tensor train_set_;
    std::vector<Image> images_;
};

} // namespace AutoStudio

#endif // OCTREE_H