
#include <string>
#include <torch/torch.h>
#include "sampler.h"
#include "../../utils/GlobalData.h"

namespace AutoStudio{

using Tensor = torch::Tensor;

AutoStudio::Sampler::Sampler(GlobalData* global_data):
    global_data_(global_data)
{
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
    Sampler* sampler;
    if (ray_sample_mode_ == 0){
        ImageSampler img_sampler = ImageSampler(global_data_);
        sampler = &img_sampler;
    }
    else if(ray_sample_mode_ == 1){
        RaySampler ray_sampler = RaySampler(global_data_);
        sampler = &ray_sampler;      
    }
    else{
        std::cout << "Not Exist Correct Object Sampler" << std::endl;
    }
    return sampler;
}


AutoStudio::ImageSampler::ImageSampler(GlobalData* global_data):Sampler(global_data)
{
    
}

AutoStudio::RaySampler::RaySampler(GlobalData* global_data):Sampler(global_data)
{

}

} //namespace AutoStudio
