/**
* This file is part of autostudio
* Copyright (C) 
**/


#ifndef FIELDSFACTORY_H
#define FIELDSFACTORY_H
#include "../../common/include/FieldModel.h"

namespace AutoStudio
{

class FieldsFactory 
{
private:
    /* data */
public:

    FieldsFactory(GlobalData* global_data);
    enum HashDType {
        OctreeMap,
        SSFNGP,
        NGP,
    };  

    std::unique_ptr<FieldModel> CreateField();
    
    HashDType hash_dtype_;
    GlobalData* global_data_;
};

} // namespace AutoStudio

#endif // FIELDSFACTORY_H