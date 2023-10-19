/**
* This file is part of autostudio
* Copyright (C) 
**/

#pragma once
#include <torch/torch.h>
#include "Eigen/Eigen"
#include "../../../Common.h"
#include "../../../dataset/Dataset.h"

namespace AutoStudio
{
using Tensor = torch::Tensor;

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
    inline void AddTreeNode(int u, int depth, Wec3f center, float bbox_len);
    std::vector<int> GetVaildCams(float bbox_len, const Tensor& center);

    int max_depth_;
    Tensor c2w_, w2c_, intri_, bound_;
    float bbox_len_;
    float dist_thres_;

    std::vector<OctreeNode> octree_nodes_;
    Tensor train_set_;
    std::vector<Image> images_;
};

} // namespace AutoStudio