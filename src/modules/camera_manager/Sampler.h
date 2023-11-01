/**
* This file is part of auto_studio
* Copyright (C) 
*  @file   camera.h
*  @author LuChen, 
*  @brief 
*/


#ifndef SAMPLER_H
#define SAMPLER_H
#include <string>
#include <torch/torch.h>
#include "Image.h"
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
    Sampler* GetInstance(std::vector<Image> images, Tensor train_set);
    virtual std::tuple<RangeRays, Tensor> GetTrainRays();

    enum RaySampleMode {
        SINGLE_IMAGE,
        ALL_IMAGES,
    };

public:
    GlobalData* global_data_;
    RaySampleMode ray_sample_mode_;
    int32_t batch_size_;
    Tensor train_set_;
    std::vector<Image> images_;
};


class ImageSampler: public Sampler
{
public:
    ImageSampler(GlobalData* global_data);
    std::tuple<RangeRays, Tensor> GetTrainRays() override;
};

class RaySampler: public Sampler
{
public:
    RaySampler(GlobalData* global_data);

    void GenAllRays();
    void GenRandRaysIdx();

    std::tuple<RangeRays, Tensor> GetTrainRays() override;

    int64_t n_rays_;
    int64_t cur_idx_;
    Tensor rgbs_, rays_o_, rays_d_, ranges_;
    Tensor rays_idx_;
};

} //namespace AutoStudio

#endif // SAMPLER_H