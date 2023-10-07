#include <torch/torch.h>

#include "Dataset.h"
#include "../common.h"
#include "../utils/Image.h"
#include "../modules/camera_manager/camera.h"

#include <iostream>
using namespace std;


namespace AutoStudio
{

Dataset::Dataset(GlobalData *global_data):
    global_data_(global_data)
{
  // TO DO: add print info
  const auto& config = global_data_->config_["dataset"];
  const auto data_path = config["data_path"].as<std::string>();
  const auto factor = config["factor"].as<float>();

  const auto& cam_list = config["cam_list"];
  int n_cameras_ = cam_list.size();

  std::vector<AutoStudio::Camera> cameras;
  std::vector<Tensor> images, poses, intrinsics;
  std::vector<std::vector<int>> train_set, val_set, test_set;

  {
    for (int i = 0; i < n_cameras_; ++i)
    {
      std::string cam_name = cam_list[i].as<std::string>();
      std::cout<< cam_name <<std::endl;
      AutoStudio::Camera camera = AutoStudio::Camera(data_path, cam_name, factor);
      set_shift(camera.train_set_,  n_images_);
      set_shift(camera.test_set_, n_images_);
      train_set.push_back(camera.train_set_);
      test_set.push_back(camera.test_set_);

    //   std::cout << camera.test_set_ <<std::endl;
      
      n_images_ = n_images_ + camera.n_images_;
      images.push_back(camera.images_tensor_);
      poses.push_back(camera.poses_);
      intrinsics.push_back(camera.intrinsics_);
      
      // TO DO: add print info for image loading
      cameras.push_back(camera);
      std::cout << n_images_ << std::endl;    
    }
    images_ = torch::stack(images, 0).contiguous();
    poses_ = torch::stack(poses, 0).contiguous().reshape({-1, 4, 4});
    intrinsics_ = torch::stack(intrinsics, 0).contiguous();

    std::cout << poses_.sizes() << std::endl;
  }


  // Normalize camera poses under a unit
  Normalize();
  

  // initial rays sample
  // generate all rays
  std::vector<Tensor> all_rays_o, all_rays_d, all_bounds;
  for (int i = 0; i < n_cameras_; ++i)
  { 
    if(i == 0){
      AutoStudio::Camera camera = cameras[i];
      auto all_rays = camera.AllRaysGenerator();
      std::cout << all_rays.origins.sizes() << std::endl;
    }
  }
  
}

void Dataset::Normalize()
{ 
 
  const auto& config = global_data_->config_["dataset"];
  Tensor cam_pos = poses_.index({Slc(), Slc(0, 3), 3}).clone();
  center_ = cam_pos.mean(0, false);
  Tensor bias = cam_pos - center_.unsqueeze(0);
  // std::cout << bias << std::endl;
  radius_ = torch::linalg_norm(bias, 2, -1, false).max().item<float>();
  // std::cout << radius_ << std::endl;
  cam_pos = (cam_pos - center_.unsqueeze(0)) / radius_;
  // std::cout << cam_pos.sizes() << std::endl;

  poses_.index_put_({Slc(), Slc(0, 3), 3}, cam_pos);
  // std::cout << poses_.sizes() << std::endl;

  poses_ = poses_.contiguous();
  c2w_ = poses_.clone();
  // std::cout << c2w_.sizes() << std::endl;
  w2c_ = torch::eye(4, CUDAFloat).unsqueeze(0).repeat({n_images_, 1, 1}).contiguous();
  w2c_.index_put_({Slc(), Slc(0, 4), Slc()}, c2w_.clone());
  w2c_ = torch::linalg_inv(w2c_);
  w2c_ = w2c_.index({Slc(), Slc(0, 4), Slc()}).contiguous();
  // std::cout << w2c_ << std::endl;
//   bounds_ = (bounds_ / radius_).contiguous();

//   Utils::TensorExportPCD(global_data_pool_->base_exp_dir_ + "/cam_pos.ply", poses_.index({Slc(), Slc(0, 3), 3}));
}

void Dataset::set_shift(std::vector<int>& set, const int shift_const)
{
    for_each(set.begin(), set.end(), [&](int& elem){ elem = elem + shift_const;});
}

} // namespace AutoStudio