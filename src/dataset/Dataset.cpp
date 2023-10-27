/**
* This file is part of autostudio
* Copyright (C) 
**/


#include <iostream>
#include <torch/torch.h>

#include "Dataset.h"
#include "../Common.h"
#include "../modules/camera_manager/Image.h"
#include "../modules/camera_manager/Camera.h"
#include "../modules/camera_manager/Sampler.h"


namespace AutoStudio
{

using Tensor = torch::Tensor;

Dataset::Dataset(GlobalData *global_data):
    global_data_(global_data)
{
  global_data_->dataset_ = reinterpret_cast<void*>(this);
  // TO DO: add print info
  const auto& config = global_data_->config_["dataset"];
  const auto data_path = config["data_path"].as<std::string>();
  const auto factor = config["factor"].as<float>();
  const auto ray_sample_mode = config["ray_sample_mode"].as<std::string>();
  const auto& cam_list = config["cam_list"];
  
  n_camera_ = cam_list.size();
  set_name_ = global_data_->config_["dataset_name"].as<std::string>();
  set_sequnceid_ = global_data_->config_["task_name"].as<std::string>();

  std::vector<Tensor> poses, intrinsics;
  std::vector<std::vector<int>> train_set, val_set, test_set;
  std::vector<std::vector<std::string>> images_fnames;
  std::vector<std::string> new_images_fnames;
  
  {
    for (int i = 0; i < n_camera_; ++i)
    {
      std::string cam_name = cam_list[i].as<std::string>();
      std::cout<< cam_name <<std::endl;
      
      // load data order by camera 
      AutoStudio::Camera camera = AutoStudio::Camera(data_path, cam_name, factor);
      
      // make data id under full space
      Set_Shift(camera.train_set_,  n_images_);
      Set_Shift(camera.test_set_, n_images_);
      train_set.push_back(camera.train_set_);
      test_set.push_back(camera.test_set_);

      n_images_ = n_images_ + camera.n_images_;
      cameras_.push_back(camera);
    }
  }
  // Normalize camera poses under a unit
  Normalize();
  
  // initial rays sampler
  auto sampler = std::make_unique<AutoStudio::Sampler>(global_data);
  
  auto images = GetFullImage<Image, Image>();
  // auto train_set_1d = Convert2DVec1D<int, int>(train_set);
  auto train_set_1d = Flatten2DVector(train_set);
  Tensor train_set_tensor = torch::from_blob(train_set_1d.data(), {int(train_set_1d.size())}, OptionInt32);
  train_set_ = train_set_tensor.to(torch::kLong).contiguous();

  sampler_ = sampler->GetInstance(images, train_set_.to(torch::kCUDA));
  // auto [train_rays, train_rgbs] = sampler_->GetTrainRays();
  // std::cout << train_rays.origins.sizes() << std::endl;
}

void Dataset::Normalize()
{
  auto poses = GetFullC2W_Tensor(true);
  const auto& config = global_data_->config_["dataset"];
  Tensor cam_pos = poses.index({Slc(), Slc(0, 3), 3}).clone();
  center_ = cam_pos.mean(0, false);
  Tensor bias = cam_pos - center_.unsqueeze(0);
  radius_ = torch::linalg_norm(bias, 2, -1, false).max().item<float>();
  cam_pos = (cam_pos - center_.unsqueeze(0)) / radius_;
  poses.index_put_({Slc(), Slc(0, 3), 3}, cam_pos);

  poses_ = poses.contiguous();
  c2w_ = poses_.clone();
  // std::cout << c2w_.sizes() << std::endl;
  w2c_ = torch::eye(4, CUDAFloat).unsqueeze(0).repeat({n_images_, 1, 1}).contiguous();
  w2c_.index_put_({Slc(), Slc(0, 3), Slc()}, c2w_.clone());
  w2c_ = torch::linalg_inv(w2c_);
  w2c_ = w2c_.index({Slc(), Slc(0, 3), Slc()}).contiguous();
//   bounds_ = (bounds_ / radius_).contiguous();

//   Utils::TensorExportPCD(global_data_pool_->base_exp_dir_ + "/cam_pos.ply", poses_.index({Slc(), Slc(0, 3), 3}));
}



Tensor Dataset::GetFullC2W_Tensor(bool device)
{
  std::vector<Tensor> c2ws;
  for (int i = 0; i < n_camera_; ++i) {
    auto camera = cameras_[i];
    auto images = camera.images_;
    for (int j = 0; j < camera.n_images_; ++j) {
      c2ws.push_back(images[j].c2w_);
    }
  }
  
  Tensor c2ws_tensor = torch::stack(c2ws, 0).reshape({-1, 3, 4});
  if (device == 1) c2ws_tensor = c2ws_tensor.to(torch::kFloat32).to(torch::kCUDA).contiguous();
  else c2ws_tensor = c2ws_tensor.to(torch::kFloat32).contiguous();
  return c2ws_tensor;
}

Tensor Dataset::GetFullW2C_Tensor(bool device)
{
  std::vector<Tensor> w2cs;
  for (int i = 0; i < n_camera_; ++i) {
    auto camera = cameras_[i];
    auto images = camera.images_;
    for (int j = 0; j < camera.n_images_; ++j) {
      Tensor w2c = torch::eye(4, CUDAFloat).contiguous();
      w2c.index_put_({Slc(0, 3), Slc()}, images[j].c2w_.clone());
      w2c = torch::linalg_inv(w2c);
      w2c = w2c.index({Slc(0, 3), Slc()}).contiguous();
      w2cs.push_back(w2c);
    }
  }
  Tensor w2cs_tensor = torch::stack(w2cs, 0).reshape({-1, 3, 4});
  if (device == 1) w2cs_tensor = w2cs_tensor.to(torch::kFloat32).to(torch::kCUDA).contiguous();
  else w2cs_tensor = w2cs_tensor.to(torch::kFloat32).contiguous();
  return w2cs_tensor;
}


Tensor Dataset::GetFullIntri_Tensor(bool device)
{
  std::vector<Tensor> intris;
  for (int i = 0; i < n_camera_; ++i) {
    auto camera = cameras_[i];
    auto images = camera.images_;
    for (int j = 0; j < camera.n_images_; ++j) {
      intris.push_back(images[j].intri_);
    }
  }
  Tensor intris_tensor = torch::stack(intris, 0).reshape({-1, 3, 3});
  if (device == 1) intris_tensor = intris_tensor.to(torch::kFloat32).to(torch::kCUDA).contiguous();
  else intris_tensor = intris_tensor.to(torch::kFloat32).contiguous();
  return intris_tensor;
}

Tensor Dataset::GetTrainC2W_Tensor(bool device)
{
  std::vector<Tensor> c2ws;
  for (int i = 0; i < n_camera_; ++i) {
    auto camera = cameras_[i];
    auto images = camera.images_;
    for (int j = 0; j < camera.n_images_; ++j) {
      c2ws.push_back(images[j].c2w_);
    }
  }
  
  Tensor c2ws_tensor = torch::stack(c2ws, 0).reshape({-1, 3, 4});
  Tensor train_c2ws  = c2ws_tensor.index({train_set_}).contiguous();
  if (device == 1) train_c2ws = train_c2ws.to(torch::kFloat32).to(torch::kCUDA).contiguous();
  else train_c2ws = train_c2ws.to(torch::kFloat32).contiguous();
  return train_c2ws;
}

Tensor Dataset::GetTrainW2C_Tensor(bool device)
{
  std::vector<Tensor> w2cs;
  for (int i = 0; i < n_camera_; ++i) {
    auto camera = cameras_[i];
    auto images = camera.images_;
    for (int j = 0; j < camera.n_images_; ++j) {
      Tensor w2c = torch::eye(4, CUDAFloat).contiguous();
      w2c.index_put_({Slc(0, 3), Slc()}, images[j].c2w_.clone());
      w2c = torch::linalg_inv(w2c);
      w2c = w2c.index({Slc(0, 3), Slc()}).contiguous();
      w2cs.push_back(w2c);
    }
  }
  Tensor w2cs_tensor = torch::stack(w2cs, 0).reshape({-1, 3, 4});
  Tensor train_w2cs = w2cs_tensor.index({train_set_}).contiguous();
  if (device == 1) train_w2cs = train_w2cs.to(torch::kFloat32).to(torch::kCUDA).contiguous();
  else train_w2cs = train_w2cs.to(torch::kFloat32).contiguous();
  return train_w2cs;
}


Tensor Dataset::GetTrainIntri_Tensor(bool device)
{
  std::vector<Tensor> intris;
  for (int i = 0; i < n_camera_; ++i) {
    auto camera = cameras_[i];
    auto images = camera.images_;
    for (int j = 0; j < camera.n_images_; ++j) {
      intris.push_back(images[j].intri_);
    }
  }
  Tensor intris_tensor = torch::stack(intris, 0).reshape({-1, 3, 3});
  Tensor train_intris = intris_tensor.index({train_set_}).contiguous();
  if (device == 1) train_intris = train_intris.to(torch::kFloat32).to(torch::kCUDA).contiguous();
  else train_intris = train_intris.to(torch::kFloat32).contiguous();
  return train_intris;
}

template <typename T>
std::vector<T> Dataset::GetFullC2W(bool device)
{
  std::vector<T> c2ws;
  for (int i = 0; i < n_camera_; ++i) {
    auto camera = cameras_[i];
    auto images = camera.images_;
    for (int j = 0; j < camera.n_images_; ++j) {
      if (device == 1) c2ws.push_back(images[j].c2w_.to(torch::kCUDA).contiguous());
      else c2ws.push_back(images[j].c2w_);
    }
  }
  return c2ws;
}

template <typename T>
std::vector<T> Dataset::GetFullIntri(bool device)
{
    std::vector<T> intris;
    for (int i = 0; i < n_camera_; ++i) {
    auto camera = cameras_[i];
    auto images = camera.images_;
    for (int j = 0; j < camera.n_images_; ++j) {
      if (device == 1) intris.push_back(images[j].intri_.to(torch::kCUDA).contiguous());
      else intris.push_back(images[j].intri_);
    }
  }
  return intris;
}

template <typename INPUT_T, typename OUTPUT_T>
std::vector<OUTPUT_T> Dataset::GetFullImage()
{ 
  std::vector<OUTPUT_T> all_images;
  for (int i = 0; i < n_camera_; ++i) {
    auto camera = cameras_[i];
    auto images = camera.images_;
    for (int j = 0; j < camera.n_images_; ++j) {
      all_images.push_back(images[j]);
    }
  }
  return all_images;
}


// Tensor Dataset::GetFullPose_Tensor(bool device)
// {
//   std::vector<Tensor> c2ws;
//   for (int i = 0; i < n_camera_; ++i) {
//     auto camera = cameras_[i];
//     auto images = camera.images_;
//     for (int j = 0; j < camera.n_images_; ++j) {
//       c2ws.push_back(images[j].c2w_);
//     }
//   }
  
//   Tensor c2ws_tensor = torch::stack(c2ws, 0).reshape({-1, 3, 4});
//   c2ws_tensor = c2ws_tensor.to(torch::kFloat32).to(torch::kCUDA).contiguous();
//   return c2ws_tensor;
// }


// template <typename INPUT_T, typename OUTPUT_T>
// std::vector<OUTPUT_T> Dataset::Convert2DVec1D(std::vector<std::vector<INPUT_T>> vec2d)
// {
//   std::vector<INPUT_T> vec1d;
//   for (int i = 0; i < vec2d.size(); ++i) {
//     for (int j = 0; j < vec2d[0].size(); ++j) {
//       vec1d.push_back(vec2d[i][j]);
//     }
//   }
//   return vec1d;
// }

template <typename T>
std::vector<T> Dataset::Flatten2DVector(const std::vector<std::vector<T>>& vec2d)
{
  std::vector<T> vec1d;

  for (const std::vector<T>& inner_vec : vec2d) {
    vec1d.insert(vec1d.end(), inner_vec.begin(), inner_vec.end());
  }
  return vec1d;
}

void Dataset::Set_Shift(std::vector<int>& set, const int shift_const)
{
  for_each(set.begin(), set.end(), [&](int& elem){ elem = elem + shift_const;});
}

} // namespace AutoStudio