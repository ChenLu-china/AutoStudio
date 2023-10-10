/**
* This file is part of autostudio
* Copyright (C) 
**/


#ifndef SAMPLER_H
#define SAMPLER_H
#include <string>
#include <torch/torch.h>
#include "image.h"
#include "../../utils/GlobalData.h"

namespace AutoStudio
{

using Tensor = torch::Tensor;

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

    Sampler(GlobalData* global_data);
    Sampler* GetInstance();
    RangeRays GetTrainRays();

    enum RaySampleMode {
        SINGLE_IMAGE,
        ALL_IMAGES,
    };

public:
    GlobalData* global_data_;
    RaySampleMode ray_sample_mode_;
    int batch_size_;
    std::vector<int> train_set_;
    std::vector<Image> images_;
};


class ImageSampler: public Sampler
{
public:
    ImageSampler(GlobalData* global_data);
    RangeRays GetTrainRays();


};

class RaySampler:public Sampler
{
public:
    RaySampler(GlobalData* global_data);
    void GenRandRaysIdx();
    RangeRays GetTrainRays();

    int64_t n_rays_;
    int64_t cur_idx_;
    Tensor rays_o_, rays_d_;
    Tensor rays_idx_;
};

} //namespace AutoStudio

#endif // SAMPLER_H