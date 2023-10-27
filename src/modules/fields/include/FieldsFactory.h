/**
* This file is part of autostudio
* Copyright (C) 
**/

#pragma once
#include "../include/FieldModel.h"

namespace AutoStudio
{

class FieldsFactory 
{
private:
    /* data */
public:

    FieldsFactory(GlobalData* global_data);
    enum HashDType{
        OctreeMap,
        NGP,
        SSFNGP,
    };  

    std::unique_ptr<FieldModel> CreateField();
    
    HashDType hash_dtype_;
    GlobalData* global_data_;
};


} // namespace AutoStudio