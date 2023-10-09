

#include <torch/torch.h>
#include "rays.h"
// #include "../../utils/GlobalData.h"

namespace AutoStudio
{
using Tensor = torch::Tensor;

AutoStudio::RaySampler::RaySampler(GlobalData* global_data_pool)
{
    const auto& config = global_data_pool->config_["dataset"];
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

// RangeRays AutoStudio::RaySampler::Train_Rays()
    
//     if(ray_sample_mode_ == 0){
        
//     }

}