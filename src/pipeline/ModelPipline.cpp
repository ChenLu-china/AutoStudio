/**
* This file is part of autostudio
* Copyright (C) 
**/

#include "ModelPipline.h"

namespace AutoStudio
{

using Tensor = torch::Tensor;

ModelPipline::ModelPipline(GlobalData* global_data)
{   
    std::cout << "Doing ModelPipline" << std::endl;
    auto field_factor = FieldsFactory(global_data);
    field_ = field_factor.CreateField();
}

} // namespace AutoStudio