#include <iostream>
#include <torch/torch.h>

#include "Dataset.h"
#include "../Common.h"
#include "../modules/camera_manager/Image.h"
#include "../modules/camera_manager/Camera.h"
#include "../modules/camera_manager/Sampler.h"


namespace AutoStudio
{

Dataset::Dataset(GlobalData *global_data):
    global_data_(global_data)
{
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
  Tensor train_set_tensor = torch::from_blob(train_set_1d.data(), train_set_1d.size(), OptionInt32);
  train_set_tensor = train_set_tensor.contiguous();
  
  sampler_ = sampler->GetInstance(images, train_set_tensor);
  // auto [train_rays, train_rgbs] = sampler_->GetTrainRays();
  // std::cout << train_rays.origins.sizes() << std::endl;
}

void Dataset::Normalize()
{ 
  
  auto poses = GetFullPose();
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


Tensor Dataset::GetFullPose()
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
  c2ws_tensor = c2ws_tensor.to(torch::kFloat32).to(torch::kCUDA).contiguous();
  return c2ws_tensor;
}


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