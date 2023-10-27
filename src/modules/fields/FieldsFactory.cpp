/**
* This file is part of autostudio
* Copyright (C) 
**/

#include <torch/torch.h> 
#include "include/FieldsFactory.h"
#include "f2nerf/OctreeMap.h"
#include "include/HashMap.h"

namespace AutoStudio
{

using Tensor = torch::Tensor;

FieldsFactory::FieldsFactory(GlobalData* global_data){
    global_data_ = global_data;
    auto models_cfg = global_data->config_["models"];
    const auto hash_dtype = models_cfg["hashtype"].as<std::string>();

    if (hash_dtype == "OctreeMap"){
        hash_dtype_ = HashDType::OctreeMap;
    }

}

std::unique_ptr<FieldModel> FieldsFactory::CreateField()
{   
    
    if (hash_dtype_ == 0)
    {   

        return std::make_unique<AutoStudio::OctreeMap>(global_data_);
    }
    CHECK(false) << "No such FieldFactory";
}


} // namespace AutoStudio
