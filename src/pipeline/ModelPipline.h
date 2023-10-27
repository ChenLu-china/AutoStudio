/**
* This file is part of autostudio
* Copyright (C) 
**/

#pragma once
#include <torch/torch.h>
#include "../utils/GlobalData.h"
#include "../modules/common/include/BaseModel.h"
#include "../modules/fields/include/FieldsFactory.h"

namespace AutoStudio
{
using Tensor = torch::Tensor;

class ModelPipline : public BaseModel
{
private:
    /* data */
public:
    ModelPipline(GlobalData* global_data);
    std::unique_ptr<FieldModel> field_;
        
    GlobalData* global_data_;
};


} // namespace AutoStudio
