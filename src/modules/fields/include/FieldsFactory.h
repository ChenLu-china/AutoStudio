/**
* This file is part of autostudio
* Copyright (C) 
**/

#pragma once
#include "../include/FieldModel.h"

namespace AutoStudio
{
enum class HashDType{
    // type   hash grid linear initial impl
    OctreeMap,
    NGP,
    SSFNGP,
};  

class FieldsFactory 
{
private:
    /* data */
public:
    FieldsFactory(GlobalData* global_data);
    std::unique_ptr<FieldModel> CreateField();
    
};


} // namespace AutoStudio