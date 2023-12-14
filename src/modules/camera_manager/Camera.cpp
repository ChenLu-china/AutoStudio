/**
* This file is part of auto_studio
* Copyright (C) 
* @file Camera.h
* @author LuChen, 
* @brief 
*/


#include <torch/torch.h>
#include <fmt/core.h>
#include <experimental/filesystem>
#include "Camera.h"
#include "Image.h"
#include "../../utils/cnpy.h"
#include "../../Common.h"

// namespace fs = std::experimental::filesystem::v1;
#include <iostream>
using namespace std;


namespace AutoStudio
{

using Tensor = torch::Tensor;
namespace fs = std::experimental::filesystem::v1;

Camera::Camera(const std::string& dir, const std::string& name, float factor, std::vector<float> bounds_factor)
{
  base_dir_ = dir;
  cam_name_ = name;
  bounds_factor_ = bounds_factor;

  // Load camera info such as dist_params and near & far
  CHECK(fs::exists(base_dir_ + "/" + cam_name_ + "/cam_info.npy"));
  Tensor dist_params, bounds;

  cnpy::NpyArray arr = cnpy::npy_load(base_dir_ + "/" + cam_name_ + "/cam_info.npy");
  auto options = torch::TensorOptions().dtype(torch::kFloat64);  // WARN: Float64 Here!!!!!
  Tensor cam_info = torch::from_blob(arr.data<double>(), arr.num_vals, options).to(torch::kFloat32).to(torch::kCUDA);

  int n_images = arr.shape[0];
  cam_info = cam_info.reshape({n_images, 6});
  dist_params = cam_info.slice(1, 0, 4).reshape({-1, 4}).contiguous();
  bounds = cam_info.slice(1, 4, 6).reshape({-1, 2}).contiguous();

  // Load images
  std::vector<string> images_fnames;
  // std::vector<Tensor> images;
  std::ifstream image_list(base_dir_ + "/" + cam_name_ + "/image_list.txt");
  std::string image_path;
  while (std::getline(image_list, image_path)) {
    std::string dir_camera = base_dir_ + "/" + cam_name_;
    auto image = AutoStudio::Image(dir_camera, image_path, factor, n_images_);
    image.dist_param_.index_put_({Slc()}, dist_params[n_images_]);
    image.near_ = bounds.index({n_images_, 0}).item<float>() * bounds_factor_[0];
    image.far_ = bounds.index({n_images_, 1}).item<float>() * bounds_factor_[1];
    images_.push_back(image);
    n_images_ = n_images_ + 1;
    // std::cout<< images_.size()<<std::endl;
  }

  // Load train/test/val split info
  try {
    cnpy::NpyArray split_arr = cnpy::npy_load(base_dir_ + "split.npy");
    CHECK_EQ(split_arr.shape[0], n_images_);
    
    auto sp_arr_ptr = split_arr.data<unsigned char>();
    for (int i = 0; i <n_images_; ++i) {
      int st = sp_arr_ptr[i];
      split_info_.push_back(st);
      if ((st & 1) == 1) train_set_.push_back(i);
      if ((st & 2) == 2) test_set_.push_back(i);
      if ((st & 4) == 4) val_set_.push_back(i);
    }
  }
  catch(...) {
    for (int i = 0; i < n_images_; i++) {
      if (i % 8 == 0) {
        test_set_.push_back(i);
      } else {
        train_set_.push_back(i);
      }
    }
  }
  
  std::cout << fmt::format("Amount of {}'s train/test/val: {}/{}/{}",
                           cam_name_, train_set_.size(), test_set_.size(), val_set_.size()) << std::endl;
  

  // Load poses and intrisics
  // {
  //   std::vector<Tensor> poses;
  //   std::vector<Tensor> intrinsics;
  //   std::vector<Tensor> dist_params;

  //   for (int i = 0; i < n_images_; ++i){
  //     // load poses
  //     std::string filename = std::string(8 - to_string(i).length(), '0') + std::to_string(i);
  //     std::string pose_path = base_dir_ + "/" + cam_name_ + "/poses" + "/" + filename + ".npy";
  //     // std::cout << pose_path << std::endl;
  //     cnpy::NpyArray arr_pose = cnpy::npy_load(pose_path);
  //     auto options = torch::TensorOptions().dtype(torch::kFloat64);
  //     Tensor pose = torch::from_blob(arr_pose.data<double>(), arr_pose.num_vals, options).to(torch::kFloat32).to(torch::kCUDA);
  //     poses.push_back(pose);

  //     // load intrinsic
  //     std::string intrinsic_path = base_dir_ + "/" + cam_name_ + "/intrinsic" + "/" + filename + ".npy"; 
  //     cnpy::NpyArray arr_intri = cnpy::npy_load(intrinsic_path);
  //     Tensor intrinsic = torch::from_blob(arr_intri.data<double>(), arr_intri.num_vals, options).to(torch::kFloat32).to(torch::kCUDA);
  //     intrinsic = intrinsic.reshape({3, 3}).contiguous();
  //     // std::cout << intrinsic.sizes() << std::endl;
  //     intrinsic.index_put_({Slc(0, 2), Slc(0, 3)}, intrinsic.index({Slc(0, 2), Slc(0, 3)}) / factor);
  //     intrinsics.push_back(intrinsic);
      
  //     // Todo:  input this from file
  //     Tensor dist_param = torch::zeros({4}, options).to(torch::kFloat32).to(torch::kCUDA);
  //     dist_params.push_back(dist_param);
  //   }
  //   poses_ = torch::stack(poses, 0).reshape({-1, 16}).contiguous();
  //   poses_ = poses_.slice(1, 0, 12).reshape({-1, 3, 4}).contiguous();
  //   poses_ = poses_.to(torch::kCUDA).contiguous();

  //   intrinsics_ = torch::stack(intrinsics, 0).reshape({-1, 3, 3}).contiguous();
  //   intrinsics_ = intrinsics_.to(torch::kCUDA).contiguous();
  //   dist_params_ = torch::stack(dist_params, 0).to(torch::kCUDA).contiguous();
  // }

  // std::cout << train_set << std::endl;
  // auto options = torch::TensorOptions().dtype(torch::kInt32);
  // train_set_ = torch::from_blob(train_set.data(), train_set.size(), options).contiguous();
  // std::cout << test_set_ << std::endl;
  
}

// RangeRays AutoStudio::Camera::AllRaysGenerator()
// { 
//   auto options = torch::TensorOptions().dtype(torch::kInt32);
//   // Tensor img_idx = torch::from_blob(img_idx_.data(), img_idx_.size(), options).contiguous().to(torch::kCUDA);
//   int H = height_;
//   int W = width_;
//   Tensor ii = torch::linspace(0.f, H - 1.f, H, CUDAFloat);
//   Tensor jj = torch::linspace(0.f, W - 1.f, W, CUDAFloat);
//   auto ij = torch::meshgrid({ii, jj}, "ij");

//   // auto ij = torch::meshgrid({ ii, jj }, "ij");
//   Tensor i = ij[0].reshape({-1});
//   Tensor j = ij[1].reshape({-1});
//   Tensor ij_ = torch::stack({i, j}, -1).to(torch::kCUDA).contiguous();
//   // std::cout << ij_.sizes() << std::endl;
//   // float near = bounds_.index({idx, 0}).item<float>();
//   // float far  = bounds_.index({idx, 1}).item<float>();
//   float near = 0.f;
//   float far = 40.f;
  
//   Tensor bound_ = torch::stack({
//                                 torch::full({H * W}, near, CUDAFloat),
//                                 torch::full({H * W}, far, CUDAFloat)
//                                 }, -1).contiguous();
  
//   std::vector<Tensor> all_rays_o, all_rays_d;
  
//   for (int k = 0; k < n_images_; ++k){
//     Tensor img_idx = torch::ones_like(i, options).to(torch::kCUDA).contiguous() * k;
//     auto [rays_o, rays_d] = Img2WorldRayFlex(img_idx, ij_.to(torch::kInt32));
//     all_rays_o.push_back(rays_o);
//     all_rays_d.push_back(rays_d);
//   }
  
//   Tensor rays_o_ = torch::stack(all_rays_o, 0).to(torch::kCUDA).contiguous();
//   Tensor rays_d_ = torch::stack(all_rays_d, 0).to(torch::kCUDA).contiguous();
//   return {rays_o_, rays_d_, bound_};
// }

} // namespace AutoStudio