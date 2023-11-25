/**
* This file is part of autostudio
* Copyright (C) 
**/

#include "ModelPipline.h"
#include "../Common.h"

namespace AutoStudio
{

using Tensor = torch::Tensor;

ModelPipline::ModelPipline(GlobalData* global_data, int n_images)
{   
    global_data_ = global_data;
    std::cout << "Doing ModelPipline" << std::endl;
    auto conf = global_data->config_["models"];
    auto field_factor = FieldsFactory(global_data);
    
    // field construction
    field_ = field_factor.CreateField();

    std::cout << "sub_models size is " << sub_models_.size() << std::endl;
    // sh construction

}

std::vector<torch::optim::OptimizerParamGroup> ModelPipline::OptimParamGroups()
{
    return field_->OptimParamGroups();
}

} // namespace AutoStudio   