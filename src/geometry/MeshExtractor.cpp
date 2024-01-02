/**
* This file is part of auto_studio
* Copyright (C) 
*  @file MeshExtractor.cpp
*  @author LuChen, ZhenJun Zhao
*  @brief 
*/
#include <torch/torch.h>

#include "MeshExtractor.h"
#include "../utils/CustomOps/CustomOps.h"
#include "../utils/CustomOps/FlexOps.h"


namespace AutoStudio
{
namespace F = torch::nn::functional;
using Tensor = torch::Tensor;

MeshExtractor::MeshExtractor(GlobalData* global_data)
{
    global_data_ = global_data;
    auto conf = global_data_->config_["mesh"];
}

void MeshExtractor::GetDensities(FieldModel* field)
{
    CHECK(false) << "Not implemented";
    return;
}


VisionMeshExtractor::VisionMeshExtractor(GlobalData* global_data): MeshExtractor(global_data)
{
    
    std::cout << "VisionMeshExtractor::VisionMeshExtractor" << std::endl;
    global_data_ = global_data;
    auto dataset = RE_INTER(Dataset*, global_data->dataset_);
    images_ = dataset->sampler_->images_;
    std::cout << "images size is:" << images_.size() <<std::endl;
    num_images_ = 1;
    // num_images_ = images_.size();
}


void VisionMeshExtractor::GetDensities(FieldModel* field)
{
    torch::NoGradGuard no_grad_guard;
    std::vector<Tensor> full_densities;
    for (int i = 0; i < num_images_; ++i){
        auto image = images_[i];
        
        image.toCUDA();
        auto [rays_o, rays_d] = image.GenRaysTensor();
        image.toHost();

        Tensor ranges = torch::stack({
                torch::full({ image.height_ * image.width_ }, image.near_, CUDAFloat),
                torch::full({ image.height_ * image.width_ }, image.far_,  CUDAFloat)}, 
                -1).contiguous();
        
        rays_o = rays_o.to(torch::kCPU);
        rays_d = rays_d.to(torch::kCPU);
        ranges = ranges.to(torch::kCPU);

        const int n_rays = rays_d.sizes()[0]; 
        const int ray_batch_size = 8192;
        for (int j = 0; j < n_rays; j += ray_batch_size){
            int i_high = std::min(j + ray_batch_size, n_rays);
            Tensor cur_rays_o = rays_o.index({Slc(j, i_high)}).to(torch::kCUDA).contiguous();
            Tensor cur_rays_d = rays_d.index({Slc(j, i_high)}).to(torch::kCUDA).contiguous();
            Tensor cur_ranges = ranges.index({Slc(j, i_high)}).to(torch::kCUDA).contiguous();

            Tensor densities = field->GetVisDensities(cur_rays_o, cur_rays_d, cur_ranges);
            std::cout << densities.sizes() << std::endl;
            full_densities.push_back(densities);
            
            // std::cout << sample_result.pts.sizes() << std::endl;
        }
        // std::cout << i << c2w << std::endl;
    }
    Tensor densities_tensor = torch::concat(full_densities, 0).contiguous();
    std::cout << densities_tensor.sizes() << std::endl;
    std::cout << "Get Density End........" << std::endl;
    return;
}


OccMeshExtractor::OccMeshExtractor(GlobalData* global_data): MeshExtractor(global_data)
{
    std::cout << "OccMeshExtractor::OccMeshExtractor" << std::endl;
    global_data_ = global_data;
}

// SampleResultFlex MeshExtractor::GetGridSamples(const Tensor& rays_o_raw, const Tensor& rays_d_raw, const Tensor& bounds_raw)
// {
    
// }

} // namespace AutoStudio
