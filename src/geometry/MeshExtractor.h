/**
* This file is part of auto_studio
* Copyright (C) 
*  @file MeshExtractor.h  
*  @author LuChen, ZhenJun Zhao
*  @brief 
*/
#ifndef MESHEXTRACTOR_H
#define MESHEXTRACTOR_H
#include <torch/torch.h>

#include "../Common.h"
#include "../dataset/Dataset.h"
#include "../utils/GlobalData.h"
#include "../modules/common/include/FieldModel.h"


namespace AutoStudio
{


class MeshExtractor
{
public:
    MeshExtractor(GlobalData* global_data);
    virtual void GetDensities(FieldModel* field);
    // virtual SampleResultFlex GetGridSamples(const Tensor& rays_o_raw, const Tensor& rays_d_raw, const Tensor& bounds_raw);

    GlobalData* global_data_;
};

class VisionMeshExtractor: public MeshExtractor
{
public:
    VisionMeshExtractor(GlobalData* global_data);
    void GetDensities(FieldModel* field) override;
    SampleResultFlex GetSamples(const Tensor& rays_o, const Tensor& rays_d, const Tensor& bounds);
    
    int num_images_;
    std::vector<Image> images_;
};

class OccMeshExtractor: public MeshExtractor
{
public:
    OccMeshExtractor(GlobalData* global_data);
};

} // namespace AutoStudio
#endif // MESHEXTRACTOR_H