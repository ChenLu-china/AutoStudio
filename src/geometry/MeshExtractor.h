/**
* This file is part of auto_studio
* Copyright (C) 
*  @file MeshExtractor.h  
*  @author LuChen, ZhenJun Zhao
*  @brief 
*/

#include "../modules/common/include/FieldModel.h"

namespace AutoStudio
{


class MeshExtractor
{
private:
    /* data */
public:
    MeshExtractor(FieldModel* model);
    SampleResultFlex GetGridSamples(const Tensor& rays_o_raw, const Tensor& rays_d_raw, const Tensor& bounds_raw);

};
 
} // namespace AutoStudio
