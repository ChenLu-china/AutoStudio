/**
* This file is part of autostudio
* Copyright (C) 
*  @file   
*  @author 
*  @brief 
*/


#include <torch/torch.h> 
#include "include/FieldsFactory.h"
#include "ngp/NGPMap.h"
#include "f2nerf/OctreeMap.h"
#include "streetsurf/SSFNGPMap.h"
#include "include/HashMap.h"


namespace AutoStudio
{

using Tensor = torch::Tensor;

FieldsFactory::FieldsFactory(GlobalData* global_data)
{
    global_data_ = global_data;
    auto models_cfg = global_data->config_["models"];
    const auto hash_dtype = models_cfg["hashtype"].as<std::string>();

    if (hash_dtype == "OctreeMap") {
        hash_dtype_ = HashDType::OctreeMap;
    }
    if (hash_dtype == "SSFNGPMap") {
        hash_dtype_ = HashDType::SSFNGP;
    }
    if (hash_dtype == "NGP") {
        hash_dtype_ = HashDType::NGP;
    }

}

std::unique_ptr<FieldModel> FieldsFactory::CreateField()
{   
    if (hash_dtype_ == 0) {   
        return std::make_unique<AutoStudio::OctreeMap>(global_data_);
    } else if(hash_dtype_ == 1) {
        return std::make_unique<AutoStudio::SSFNGPMap>(global_data_);
    }else if (hash_dtype_ == 2) {
        return std::make_unique<AutoStudio::NGPMap>(global_data_);
    } else {
        CHECK(false) << "No such FieldFactory";     
    }
    return nullptr;
}

} // namespace AutoStudio
