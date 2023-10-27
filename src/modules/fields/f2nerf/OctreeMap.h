/**
* This file is part of autostudio
* Copyright (C) 
**/
#pragma once
#include <torch/torch.h>
#include "Octree.h"
#include "../include/FieldModel.h"

namespace AutoStudio
{

class OctMap : public FieldModel
{
public:
    OctMap(GlobalData* global_data);

    std::unique_ptr<Octree> octree_;
};

} // namespace AutoStudio