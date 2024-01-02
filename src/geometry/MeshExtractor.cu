/**
* This file is part of auto_studio
* Copyright (C) 
*  @file MeshExtractor.h  
*  @author LuChen, ZhenJun Zhao
*  @brief 
*/

#include "MeshExtractor.h"

namespace AutoStudio
{

using Tensor = torch::Tensor;

SampleResultFlex VisionMeshExtractor::GetSamples(const Tensor& rays_o_raw, const Tensor& rays_d_raw, const Tensor& bounds_raw)
{
    Tensor rays_o = rays_o_raw.contiguous();
    Tensor rays_d = (rays_d_raw / torch::linalg_norm(rays_d_raw, 2, -1, true)).contiguous();
    
    int n_rays = rays_o.sizes()[0];
    Tensor bounds = torch::stack({torch::full({n_rays}, global_data_->near_)});

    return {Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
}

} // namespace AutoStudio
