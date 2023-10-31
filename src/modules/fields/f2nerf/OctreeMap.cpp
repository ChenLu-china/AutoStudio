/**
* This file is part of autostudio
* Copyright (C) 
**/
#include <torch/torch.h>
#include "OctreeMap.h"
#include "../include/FieldModel.h"

namespace AutoStudio
{
using Tensor = torch::Tensor;

OctreeMap::OctreeMap(GlobalData* global_data)
{
    global_data_ = global_data;
    auto config = global_data_->config_["models"];
    auto dataset = RE_INTER(Dataset*, global_data_->dataset_);

    float split_dist_thres = config["split_dist_thres"].as<float>();
    int max_level = config["max_level"].as<int>();
    int bbox_levels = config["bbox_levels"].as<int>();
    float bbox_side_len = (1 << (bbox_levels - 1));
    
    octree_ = std::make_unique<Octree>(max_level, bbox_side_len, split_dist_thres, dataset);
    
}

std::vector<Tensor> OctreeMap::States()
{
    std::vector<Tensor> ret;
    ret.push_back(octree_->octree_nodes_gpu_);
    ret.push_back(octree_->octree_trans_gpu_);
    ret.push_back(octree_->tree_visit_cnt_);
    Tensor milestones_ts = torch::from_blob(sub_div_milestones_.data(), sub_div_milestones_.size(), CPUInt).to(torch::kCUDA);
    ret.push_back(milestones_ts);
    
    return ret;
}

int OctreeMap::LoadStates(const std::vector<Tensor>& states, int idx)
{
    octree_->octree_nodes_gpu_ = states[idx++].clone().to(torch::kCUDA).contiguous();
    octree_->octree_trans_gpu_ = states[idx++].clone().to(torch::kCUDA).contiguous();
    octree_->tree_visit_cnt_   = states[idx++].clone().to(torch::kCUDA).contiguous();
    Tensor milestones_ts       = states[idx++].clone().to(torch::kCUDA).contiguous();
    
    Tensor octree_node_cpu = octree_ ->octree_nodes_gpu_.to(torch::kCPU);
    octree_->octree_nodes_.resize(octree_node_cpu.sizes()[0] / sizeof(OctreeNode));
    std::memcpy(octree_->octree_nodes_.data(), octree_node_cpu.data_ptr(), octree_node_cpu.sizes()[0]);

    Tensor octree_trans_cpu = octree_->octree_trans_gpu_.to(torch::kCPU);
    octree_->octree_trans_.resize(octree_trans_cpu.sizes()[0] / sizeof(OctreeTransInfo));
    std::memcpy(octree_->octree_trans_.data(), octree_trans_cpu.data_ptr(), octree_trans_cpu.sizes()[0]);

    sub_div_milestones_.resize(milestones_ts.sizes()[0]);
    std::memcpy(sub_div_milestones_.data(), milestones_ts.data_ptr(), milestones_ts.sizes()[0] * sizeof(int));
    PRINT_VAL(sub_div_milestones_);

    octree_->tree_weight_stats_ = torch::full({ int(octree_->octree_nodes_.size()) }, INIT_NODE_STAT, CUDAInt);
    octree_->tree_alpha_stats_ = torch::full({ int(octree_->octree_nodes_.size()) }, INIT_NODE_STAT, CUDAInt);

    int valid_nodes = 0;
    for(int i = 0; i < octree_->octree_nodes_.size(); ++i){
        if (octree_->octree_nodes_[i].trans_idx_ >= 0){
            valid_nodes++;
        }
    }
    PRINT_VAL(valid_nodes);

    return idx;
}

} // namespace AutoStudio