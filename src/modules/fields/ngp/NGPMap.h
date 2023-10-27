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
class NGPMap : public FieldModel
{
private:
    /* data */
public:
    ONGPMap(GlobalData* global_data);
};


} // namespace AutoStudio