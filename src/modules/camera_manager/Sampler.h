/**
* This file is part of auto_studio
* Copyright (C) 
* @file   Sampler.h
* @author LuChen, 
* @brief 
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

struct alignas(32) RangeRays {
    Tensor origins;
    Tensor dirs;
    Tensor ranges;
};


class Sampler
{
public: 

    Sampler(GlobalData* global_data);
    Sampler* GetInstance(std::vector<Image> images, Tensor train_set, Tensor test_set);
    virtual std::tuple<RangeRays, Tensor> TestRays(int& vis_idx);
    virtual std::tuple<RangeRays, Tensor, Tensor> GetTrainRays();
    virtual std::tuple<int, int> Get_HW(int& vis_idx);

    enum RaySampleMode {
        SINGLE_IMAGE,
        ALL_IMAGES,
        MULTI_IMAGES,
    };

public:
    GlobalData* global_data_;
    RaySampleMode ray_sample_mode_;
    int32_t batch_size_;
    Tensor train_set_;
    Tensor test_set_;
    std::vector<Image> images_;
};


class ImageSampler: public Sampler
{
public:
    ImageSampler(GlobalData* global_data);

    std::tuple<RangeRays, Tensor, Tensor> GetTrainRays() override;
    std::tuple<int, int> Get_HW(int& vis_idx) override;
    // std::tuple<>
};

class RaySampler: public Sampler
{
public:
    RaySampler(GlobalData* global_data);

    void GenAllRays();
    void GenRandRaysIdx();

    std::tuple<RangeRays, Tensor, Tensor> GetTrainRays() override;
    

    int64_t n_rays_;
    int64_t cur_idx_;
    Tensor rgbs_, rays_o_, rays_d_, ranges_;
    Tensor rays_idx_;
};

class OriSampler: public Sampler
{
public: 
    OriSampler(GlobalData* global_data);
    void GatherData();
    std::tuple<RangeRays, Tensor, Tensor> GetTrainRays() override;
    std::tuple<Tensor, Tensor> Img2WorldRayFlex(const Tensor& cam_indices, const Tensor& ij);
    
    int width_, height_;
    Tensor image_tensors_, poses_, intri_, dist_params_, ranges_;
};

} //namespace AutoStudio

#endif // SAMPLER_H