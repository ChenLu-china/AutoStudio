/**
* This file is part of autostudio
* Copyright (C) 
**/

#include <torch/torch.h> 
#include "include/FieldsFactory.h"
#include "include/HashMap.h"

namespace AutoStudio
{
using Tensor = torch::Tensor;

FieldsFactory::FieldsFactory(GlobalData* global_data){
    auto models_cfg = global_data->config_["models"];
}

} // namespace AutoStudio
