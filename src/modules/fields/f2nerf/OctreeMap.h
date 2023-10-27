/**
* This file is part of autostudio
* Copyright (C) 
**/


#ifndef OCTREEMAP_H
#define OCTREEMAP_H
#include <torch/torch.h>
#include "Octree.h"
#include "../include/FieldModel.h"


namespace AutoStudio
{

class OctreeMap : public FieldModel
{
public:
    OctreeMap(GlobalData* global_data);

    std::unique_ptr<Octree> octree_;
};

} // namespace AutoStudio

#endif // OCTREEMAP_H