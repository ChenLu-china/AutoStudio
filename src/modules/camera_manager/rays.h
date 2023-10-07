#pragma once

#include <string>
#include <torch/torch.h>
#include "../../utils/GlobalData.h"

namespace AutoStudio
{

struct alignas(32) Rays{
    torch::Tensor origins;
    torch::Tensor dirs;
};

struct alignas(32) RangeRays{
    torch::Tensor origins;
    torch::Tensor dirs;
    torch::Tensor ranges;
};


class RaySampler{

public:
    RaySampler(GlobalData* global_data_pool);

    enum RaySampleMode {
        SINGLE_IMAGE, ALL_IMAGES,
    };

    //Rays
    RangeRays rays ;

    RaySampleMode ray_sample_mode_;
};




} // AutoStudio