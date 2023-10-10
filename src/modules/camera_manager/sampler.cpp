
#include <string>
#include <torch/torch.h>
#include "sampler.h"
#include "../../utils/GlobalData.h"

namespace AutoStudio
{

using Tensor = torch::Tensor;

Sampler::Sampler(GlobalData* global_data):
    global_data_(global_data)
{
    // global_data_ = global_data;
    const auto& config = global_data_->config_["dataset"];
    const auto& batch_size = config["batch_size"].as<std::int8_t>();
    const auto& ray_sample_mode = config["ray_sample_mode"].as<std::string>();
    
    if (ray_sample_mode == "single_image")
    {
        ray_sample_mode_ = RaySampleMode::SINGLE_IMAGE;
    }
    else if (ray_sample_mode == "all_images")
    {
        ray_sample_mode_  = RaySampleMode::ALL_IMAGES;
    }
    else
    {
        std::cout << "Invalid Rays Sampler Mode" << std::endl;
        std::exit;
    }
    std::cout << ray_sample_mode_ << std::endl;
    
}

Sampler* Sampler::GetInstance()
{   
    // Sampler* sampler;
    if (ray_sample_mode_ == 0) {
        return new ImageSampler(global_data_);
    } else if(ray_sample_mode_ == 1) {
        return new RaySampler(global_data_);
    } else {
        std::cout << "Not Exist Correct Object Sampler" << std::endl;
        return nullptr;
    }
}


ImageSampler::ImageSampler(GlobalData* global_data):Sampler(global_data)
{
    
}

RaySampler::RaySampler(GlobalData* global_data):Sampler(global_data)
{
    int n_image = images_.size();
    std::vector<Tensor> rays_o, rays_d, ranges;
    for(int i = 0; i < n_image; ++i){
       auto image = images_[i];
       auto [ray_o, ray_d] = image.GenRaysTensor();
       rays_o.push_back(ray_o);
       rays_d.push_back(ray_d);   
    }
    Tensor rays_o_tensor = torch::stack(rays_o, 0).to(torch::kCUDA);
    Tensor rays_d_tensor = torch::stack(rays_d, 0).to(torch::kCUDA);

    rays_o_ = rays_o_tensor.to(torch::kFloat32).reshape({-1, 3}).contiguous();
    rays_d_ = rays_d_tensor.to(torch::kFloat32).reshape({-1, 3}).contiguous();

    n_rays_ = int64_t(rays_o_.size(0));
    GenRandRaysIdx();
}

RangeRays RaySampler::GetTrainRays()
{
    int n_image = images_.size();
    std::vector<Tensor> rays_o, rays_d, ranges;
    for(int i = 0; i < n_image; ++i){
       auto image = images_[i];
       auto [ray_o, ray_d] = image.GenRaysTensor();
       rays_o.push_back(ray_o);
       rays_d.push_back(ray_d);   
    }
    Tensor rays_o_tensor = torch::stack(rays_o, 0).to(torch::kCUDA);
    Tensor rays_d_tensor = torch::stack(rays_d, 0).to(torch::kCUDA);

    rays_o_ = rays_o_tensor.to(torch::kFloat32).reshape({-1, 3}).contiguous();
    rays_d_ = rays_d_tensor.to(torch::kFloat32).reshape({-1, 3}).contiguous();

    n_rays_ = int64_t(rays_o_.size(0));
    GenRandRaysIdx();
}

RangeRays RaySampler::GetTrainRays()
{

}

void AutoStudio::RaySampler::GenRandRaysIdx()
{   
    cur_idx_ = 0;
    auto option = torch::TensorOptions().dtype(torch::kLong);
    Tensor shuffled_rays_indices = torch::randperm(n_rays_, option);
    rays_idx_ = shuffled_rays_indices.contiguous();
    std::cout << rays_idx_ << std::endl;
}

} //namespace AutoStudio
