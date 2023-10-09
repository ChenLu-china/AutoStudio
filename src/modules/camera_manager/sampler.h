# pragma once

#include <string>
#include <torch/torch.h>
#include "image.h"
#include "../../utils/GlobalData.h"

namespace AutoStudio{

using Tensor = torch::Tensor;

GlobalData;

struct alignas(32) Rays{
    Tensor origins;
    Tensor dirs;
};

struct alignas(32) RangeRays{
    Tensor origins;
    Tensor dirs;
    Tensor ranges;
};

class Sampler
{  

public: 
    enum RaySampleMode {
        SINGLE_IMAGE, ALL_IMAGES,
    };

    Sampler(GlobalData* global_data);
    
public:
    int batch_size_;
    RaySampleMode ray_sample_mode_;
};


class ImageSampler:Sampler
{
public:
    ImageSampler();
    
public:
    std::vector<int> train_set_;
    std::vector<Image> images_;
};


} //namespace AutoStudio
