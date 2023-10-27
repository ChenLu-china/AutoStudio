/**
* This file is part of autostudio
* Copyright (C) 
**/

#include "OctreeMap.h"
#include "../include/FieldModel.h"

namespace AutoStudio
{

OctMap::OctMap(GlobalData* global_data){
    global_data_ = global_data;
    auto config = global_data_->config_["models"];
    auto dataset = RE_INTER(Dataset*, global_data_->dataset_);

    float split_dist_thres = config["split_dist_thres"].as<float>();
    int max_level = config["max_level"].as<int>();
    int bbox_levels = config["bbox_levels"].as<int>();
    float bbox_side_len = (1 << (bbox_levels - 1));
    
    octree_ = std::make_unique<Octree>(max_level, bbox_side_len, split_dist_thres, dataset);
}


} // namespace AutoStudio