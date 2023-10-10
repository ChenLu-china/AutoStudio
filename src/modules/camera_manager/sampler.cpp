
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

}

} //namespace AutoStudio
