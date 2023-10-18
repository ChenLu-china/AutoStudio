/**
* This file is part of autostudio
* Copyright (C) 
**/

#pragma once
#include <torch/torch.h>
#include "Eigen/Eigen"
#include "../../../Common.h"


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
    Octree(int max_depth, float bbox_side_len, float split_dist_thres,
             const Tensor& c2w, const Tensor& w2c, const Tensor& intri, const Tensor& bound);
    ~Octree();
};

Octree::Octree(int max_depth, float bbox_side_len, float split_dist_thres,
             const Tensor& c2w, const Tensor& w2c, const Tensor& intri, const Tensor& bound)
{
}

Octree::~Octree()
{
}


} // namespace AutoStudio