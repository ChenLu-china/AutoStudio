/**
* This file is part of autostudio
* Copyright (C) 
**/
#pragma once
#include <torch/torch.h>
#include "../../../Common.h"
#include "../include/FieldModel.h"
#include "../../../dataset/Dataset.h"

namespace AutoStudio
{
using Tensor = torch::Tensor;

class SSFNGPMap : public FieldModel
{
private:

public:
    SSFNGPMap(GlobalData* global_data);
};


} // namespace AutoStudio