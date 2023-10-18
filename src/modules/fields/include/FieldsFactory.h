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
    Dense,
    Hash,
    NGP,

};  

class FieldsFactory 
{
private:
    /* data */
public:
    std::unique_ptr<FieldModel> CreateField(GlobalData* global_data); 
};


} // namespace AutoStudio