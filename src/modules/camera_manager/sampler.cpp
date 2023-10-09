
#include <string>
#include <torch/torch.h>
#include "sampler.h"

namespace AutoStudio{

using Tensor = torch::Tensor;

AutoStudio::Sampler::Sampler(GlobalData* global_data)
{
    const auto& config = global_data->config_["dataset"];
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

} //namespace AutoStudio
