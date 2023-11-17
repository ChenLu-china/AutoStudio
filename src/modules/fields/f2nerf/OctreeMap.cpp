/**
* This file is part of autostudio
* Copyright (C) 
**/
#include <torch/torch.h>
#include "OctreeMap.h"

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
    
    //construct octree and make warp transformation matrix
    octree_ = std::make_unique<Octree>(max_level, bbox_side_len, split_dist_thres, dataset); 

    // visualize octree map
    VisOctree();
    global_data_ -> n_volumes_ = octree_ -> octree_trans_.size();
    std::cout << "The number of valid leaf node is :" << octree_ -> octree_trans_.size() << std::endl;
    std::cout << "The number of valid leaf node is :" << global_data_->n_volumes_ << std::endl;
    // construct network
    hashmap_ = std::make_unique<Hash3DVertex>(global_data_);


}

void OctreeMap::VisOctree()
{
    std::ofstream f(global_data_ -> base_exp_dir_ + "/octree.obj", std::ios::out);

    auto& octree_nodes = octree_->octree_nodes_;
    int n_nodes = octree_->octree_nodes_.size();
    for(const auto& node : octree_nodes){
        for(int st = 0; st < 8; ++st){
            Wec3f xyz = node.center_ + Wec3f(((st >> 2 & 1) - 0.5f), ((st >> 1 & 1) - 0.5f), ((st >> 0 & 1) - 0.5f)) * node.extend_len_;
            f << "v " << xyz[0] << " " << xyz[1] << " " << xyz[2] << std::endl;
        }
    }

    for(int i = 0; i< n_nodes; ++i){
        if (octree_nodes[i].trans_idx_ < 0) { continue; }
        for (int a = 0; a < 8; ++a){
            for(int b = a + 1; b < 8; ++b){
                int st = (a ^ b);
                if (st == 1 || st == 2 || st == 4){
                   f << "l " << i * 8 + a + 1 << " " << i * 8 + b + 1 << std::endl;
                }
            }
        }
    }

    f.close();
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