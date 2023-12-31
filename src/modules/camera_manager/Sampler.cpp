/**
* This file is part of auto_studio
* Copyright (C) 
* @file   Sampler.cpp
* @author 
* @brief 
*/


#include <string>
#include <fmt/core.h>
#include <torch/torch.h>
#include "Sampler.h"
#include "../../Common.h"
#include "../../utils/GlobalData.h"


namespace AutoStudio
{

using Tensor = torch::Tensor;

Sampler::Sampler(GlobalData* global_data):
    global_data_(global_data)
{
    // global_data_ = global_data;
    const auto& config = global_data_->config_["dataset"];
    const auto batch_size = config["batch_size"].as<std::int32_t>();
    const auto ray_sample_mode = config["ray_sample_mode"].as<std::string>();
    
    if (ray_sample_mode == "single_image") {
        ray_sample_mode_ = RaySampleMode::SINGLE_IMAGE;
    } else if (ray_sample_mode == "all_images") {
        ray_sample_mode_  = RaySampleMode::ALL_IMAGES;
    } else if (ray_sample_mode == "multi_images") {
        ray_sample_mode_  = RaySampleMode::MULTI_IMAGES;
    }
    else {
        std::cout << "Invalid Rays Sampler Mode" << std::endl;
        std::exit;
    }

    batch_size_ = batch_size;
    // std::cout << ray_sample_mode_ << std::endl; 
}

Sampler* Sampler::GetInstance(std::vector<Image> images, Tensor train_set, Tensor test_set)
{   
    // Sampler* sampler;
    if (ray_sample_mode_ == 0) {
        auto image_sampler = new ImageSampler(global_data_);
        image_sampler->images_ = images;
        image_sampler->train_set_ = train_set;
        image_sampler->test_set_ = test_set;
        return image_sampler;
    } else if(ray_sample_mode_ == 1) {
        auto ray_sampler = new RaySampler(global_data_);
        ray_sampler->images_ = images;
        ray_sampler->train_set_ = train_set;
        ray_sampler->test_set_ = test_set;
        ray_sampler->GenAllRays();
        return ray_sampler;
    } else if(ray_sample_mode_ == 2) {
        auto ray_sampler = new OriSampler(global_data_);
        ray_sampler->images_ = images;
        ray_sampler->train_set_ = train_set;
        ray_sampler->test_set_ = test_set;
        ray_sampler->GatherData();
        return ray_sampler;
    } else {
        std::cout << "Not Exist Correct Object Sampler" << std::endl;
        return nullptr;
    }
}

std::tuple<RangeRays, Tensor> Sampler::TestRays(int& vis_idx)
{
    auto image = images_[vis_idx];
    image.toCUDA();
    auto [rays_o, rays_d] = image.GenRaysTensor();
    image.toHost();

    Tensor range = torch::stack({
                torch::full({ image.height_ * image.width_ }, image.near_, CUDAFloat),
                torch::full({ image.height_ * image.width_ }, image.far_,  CUDAFloat)}, 
                -1).contiguous();
    
    rays_o = rays_o.reshape({-1, 3}).contiguous();
    rays_d = rays_d.reshape({-1, 3}).contiguous();
    Tensor rgbs = image.img_tensor_.reshape({-1, 3}).to(torch::kCUDA).contiguous();
    range = range.reshape({-1, 2});
    return {{rays_o, rays_d, range}, rgbs};
}

std::tuple<RangeRays, Tensor, Tensor> Sampler::GetTrainRays()
{
    RangeRays rays;
    Tensor rgbs;
    return {rays, rgbs, Tensor()};
}

std::tuple<int, int> Sampler::Get_HW(int& vis_idx)
{
    auto image = images_[vis_idx];
    int H = image.height_;
    int W = image.width_;
    return {H, W};
}

/**
 *  ImageSampler fucntion implementation
*/

ImageSampler::ImageSampler(GlobalData* global_data) : Sampler(global_data)
{   
    std::string set_name = global_data_->config_["dataset_name"].as<std::string>();
    fmt::print("The {} dataset use Single Image Sampler\n", set_name);
}

std::tuple<RangeRays, Tensor, Tensor> ImageSampler::GetTrainRays() 
{
    // std::cout << "Image Sampler" << std::endl;
    int cam_idx =  torch::randint(train_set_.size(0), {1}).item<int>();
    cam_idx = train_set_.index({cam_idx}).item<int>();
    // std::cout << cam_idx << std::endl;
    auto image = images_[cam_idx];
    image.toCUDA(); 
    
    auto [rays_o, rays_d] = image.GenRaysTensor();
    Tensor range = torch::stack({
                torch::full({ image.height_ * image.width_ }, image.near_, CUDAFloat),
                torch::full({ image.height_ * image.width_ }, image.far_,  CUDAFloat)}, 
                -1).contiguous();
    range = range.reshape({-1, 2});
    image.toHost();
    
    int n_rays = image.width_ * image.height_;
    Tensor shuffled_rays_indices = torch::randperm(n_rays, OptionLong).contiguous();
    Tensor sel_indices = shuffled_rays_indices.index({Slc(0, batch_size_, 1)});

    Tensor sel_rays_o = rays_o.index({sel_indices}).to(torch::kCUDA).contiguous();
    Tensor sel_rays_d = rays_d.index({sel_indices}).to(torch::kCUDA).contiguous();
    Tensor sel_ranges = range.index({sel_indices}).to(torch::kCUDA).contiguous();
    Tensor sel_rgbs = image.img_tensor_.view({-1, 3}).index({sel_indices}).to(torch::kCUDA).contiguous();
    Tensor sel_cams_idx = torch::full({batch_size_}, cam_idx, CUDAInt);
    return {{sel_rays_o, sel_rays_d, sel_ranges}, sel_rgbs, sel_cams_idx};
}

// std::tuple<RangeRays, Tensor> ImageSampler::TestRays(int& vis_idx)
// {
//     auto image = images_[vis_idx];
//     image.toCUDA();
//     auto [rays_o, rays_d] = image.GenRaysTensor();
//     image.toHost();

//     Tensor range = torch::stack({
//                 torch::full({ image.height_ * image.width_ }, image.near_, CUDAFloat),
//                 torch::full({ image.height_ * image.width_ }, image.far_,  CUDAFloat)}, 
//                 -1).contiguous();
    
//     rays_o = rays_o.reshape({-1, 3}).contiguous();
//     rays_d = rays_d.reshape({-1, 3}).contiguous();
//     Tensor rgbs = image.img_tensor_.reshape({-1, 3}).to(torch::kCUDA).contiguous();
//     range = range.reshape({-1, 2});
//     return {{rays_o, rays_d, range}, rgbs};
// }

std::tuple<int, int> ImageSampler::Get_HW(int& vis_idx)
{
    auto image = images_[vis_idx];
    int H = image.height_;
    int W = image.width_;
    return {H, W};
}

/**
 *  RaySampler fucntion implementation
*/

RaySampler::RaySampler(GlobalData* global_data):Sampler(global_data)
{
    std::string set_name = global_data_->config_["dataset_name"].as<std::string>();
    fmt::print("The {} dataset use Ray Sampler\n", set_name);
}

void RaySampler::GenAllRays()
{
    const int n_image = train_set_.sizes()[0];
    std::vector<Tensor> rgbs, rays_o, rays_d, ranges;
    for (int i = 0; i < n_image; ++i) {
        int cam_idx = train_set_.index({i}).item<int>();
        auto image = images_[cam_idx];
        image.toCUDA();
        auto [ray_o, ray_d] = image.GenRaysTensor();
        rays_o.push_back(ray_o);
        rays_d.push_back(ray_d);
        rgbs.push_back(image.img_tensor_);
        
        Tensor range = torch::stack({
                torch::full({ image.height_ * image.width_ }, image.near_, CUDAFloat),
                torch::full({ image.height_ * image.width_ }, image.far_,  CUDAFloat)}, 
                -1).contiguous();

            ranges.push_back(range); 
    }
    Tensor rays_o_tensor = torch::stack(rays_o, 0).to(torch::kCUDA);
    Tensor rays_d_tensor = torch::stack(rays_d, 0).to(torch::kCUDA);
    Tensor rgbs_tensor = torch::stack(rgbs, 0).to(torch::kCUDA);
    Tensor ranges_tensor = torch::stack(ranges, 0).to(torch::kCUDA);

    rays_o_ = rays_o_tensor.to(torch::kFloat32).reshape({-1, 3}).contiguous();
    rays_d_ = rays_d_tensor.to(torch::kFloat32).reshape({-1, 3}).contiguous();
    rgbs_ =  rgbs_tensor.to(torch::kFloat32).reshape({-1, 3}).contiguous();
    ranges_ = ranges_tensor.to(torch::kFloat32).reshape({-1, 2}).contiguous();

    n_rays_ = int64_t(rays_o_.size(0));
    GenRandRaysIdx();
}

void RaySampler::GenRandRaysIdx()
{   
    cur_idx_ = 0;
    auto option = torch::TensorOptions().dtype(torch::kLong);
    Tensor shuffled_rays_indices = torch::randperm(n_rays_, option);
    rays_idx_ = shuffled_rays_indices.contiguous();
    std::cout << rays_idx_.sizes() << std::endl;
}

std::tuple<RangeRays, Tensor, Tensor> RaySampler::GetTrainRays()
{
    int64_t end_idx = cur_idx_ + batch_size_;
    int64_t batch_size; 
    if (end_idx > n_rays_) {
        batch_size = n_rays_ - cur_idx_;
    } else {
        batch_size = batch_size_;
    } 
    
    Tensor sel_idx    = rays_idx_.index({Slc(cur_idx_, cur_idx_ + batch_size)});
    Tensor sel_rays_o = rays_o_.index({sel_idx}).contiguous();
    Tensor sel_rays_d = rays_d_.index({sel_idx}).contiguous();
    Tensor sel_rgbs   = rgbs_.index({sel_idx}).contiguous();
    Tensor sel_ranges = ranges_.index({sel_idx}).contiguous();
    
    return {{sel_rays_o, sel_rays_d, sel_ranges}, sel_rgbs, Tensor()};
}

// std::tuple<RangeRays, Tensor> RaySampler::TestRays(int& vis_idx)
// {
//     std::cout << "Test rays whether correct?" << std::endl;
//     return {{Tensor(), Tensor(), Tensor()}, Tensor()};
// }

/**
 *  OriSampler fucntion implementation
*/

OriSampler::OriSampler(GlobalData* global_data):Sampler(global_data)
{
    std::string set_name = global_data_->config_["dataset_name"].as<std::string>();
    fmt::print("The {} dataset use Original Sampler\n", set_name);
}

void OriSampler::GatherData()
{   
    std::vector<Tensor> images, poses, intris, dist_params, ranges;
    int n = int(images_.size());
    for (int i = 0; i < n; ++i) {
        images.push_back(images_[i].img_tensor_);
        poses.push_back(images_[i].c2w_);
        intris.push_back(images_[i].intri_);
        dist_params.push_back(images_[i].dist_param_);
        Tensor range  = torch::stack({
                                   torch::full({ 1 }, images_[i].near_),
                                   torch::full({ 1 }, images_[i].far_)
                               }, -1).contiguous();
        ranges.push_back(range);
        if (i == 0) {
            height_ = images_[i].height_;
            width_ = images_[i].width_;
        }
    }

    image_tensors_ = torch::stack(images, 0).reshape({-1, 3});
    poses_ = torch::stack(poses, 0).reshape({-1, 3, 4});
    intri_ = torch::stack(intris,0).reshape({-1, 3, 3});
    dist_params_ = torch::stack(dist_params, 0).reshape({-1, 4});
    ranges_ = torch::stack(ranges, 0).reshape({-1, 2});

    image_tensors_ = image_tensors_.to(torch::kCUDA).contiguous();
    poses_ = poses_.to(torch::kCUDA).contiguous();
    intri_ = intri_.to(torch::kCUDA).contiguous();
    dist_params_ = dist_params_.to(torch::kCUDA).contiguous();
    ranges_ = ranges_.to(torch::kCUDA).contiguous();

}

std::tuple<RangeRays, Tensor, Tensor> OriSampler::GetTrainRays(){
    // std::vector<int> img_idx; 
    // img_idx.insert(img_idx.end(), train_set_.begin(), train_set_.end());
    // Tensor cur_set = torch::from_blob(img_idx.data(), { int(img_idx.size())}, CPUInt);
    train_set_ = train_set_.to(CPUInt);
    int num = train_set_.sizes()[0];
    Tensor cam_indices = torch::randint(num, { batch_size_ }, CPULong); // Torch index need "long long" type
    cam_indices = train_set_.index({cam_indices}).contiguous();
    Tensor i = torch::randint(0, height_, batch_size_, CPULong);
    Tensor j = torch::randint(0, width_, batch_size_, CPULong);
    Tensor ij = torch::stack({i, j}, -1).to(torch::kCUDA).contiguous();

    Tensor gt_colors = image_tensors_.view({-1, 3}).index({ (cam_indices * height_ * width_ + i * width_ + j).to(torch::kLong) }).to(torch::kCUDA).contiguous();
    cam_indices = cam_indices.to(torch::kCUDA);
    auto [ rays_o, rays_d ] = Img2WorldRayFlex(cam_indices.to(torch::kInt32), ij.to(torch::kInt32));
    Tensor ranges = ranges_.index({cam_indices.to(torch::kLong)}).contiguous();
    return { { rays_o, rays_d, ranges }, gt_colors, cam_indices.to(torch::kInt32).contiguous() };
}

} //namespace AutoStudio
