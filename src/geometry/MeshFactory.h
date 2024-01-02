/**
* This file is part of auto_studio
* Copyright (C) 
*  @file MeshFactory.h  
*  @author LuChen, ZhenJun Zhao
*  @brief 
*/

#ifndef MESHFACTORY_H
#define MESHFACTORY_H
#include "MeshExtractor.h"
#include "../modules/common/include/FieldModel.h"
#include "../utils/GlobalData.h"

namespace AutoStudio
{


class MeshFactory
{
private:
    /* data */
public:
  
    MeshFactory(GlobalData* global_data);
    enum MeshDType {
        VIS,
        OCC,
    };
    std::unique_ptr<MeshExtractor> CreateMeshExtractor();
    
    MeshDType mesh_dtype_;
    GlobalData* global_data_;
};

} // namespace AutoStudio

#endif // MESHFACTORY_H