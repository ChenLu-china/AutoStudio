#include <torch/torch.h>

#include "Dataset.h"
#include "../common.h"
#include "../modules/camera_manager/image.h"
#include "../modules/camera_manager/camera.h"
#include "../modules/camera_manager/sampler.h"

// #include "../modules/camera_manager/rays.h"

#include <iostream>


namespace AutoStudio
{

Dataset::Dataset(GlobalData *global_data):
    global_data_(global_data)
{
  // TO DO: add print info
  const auto& config = global_data_->config_["dataset"];
  const auto data_path = config["data_path"].as<std::string>();
  const auto factor = config["factor"].as<float>();
  const auto& ray_sample_mode = config["ray_sample_mode"].as<std::string>();
  const auto& cam_list = config["cam_list"];
  n_camera_ = cam_list.size();

  std::vector<Tensor>  poses, intrinsics;
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

    //   std::cout << camera.test_set_ <<std::endl;
      n_images_ = n_images_ + camera.n_images_;
      cameras_.push_back(camera);
    }

    // images_ = torch::stack(images, 0).contiguous();
    // poses_ = torch::stack(poses, 0).contiguous().reshape({-1, 3, 4});
    // intrinsics_ = torch::stack(intrinsics, 0).contiguous();

    // convert 2D vector to 1D vector
    // new_images_fnames = Convert2DVec1D(images_fnames);
    // TO DO: add print info for image loading  
    std::cout << n_images_ << std::endl;  
    // std::cout << poses_.sizes() << std::endl;
  }

  // Normalize camera poses under a unit
  Normalize();
  
  // initial rays sampler
  // AutoStudio::Sampler ray_sampler = AutoStudio::Sampler(global_data_);
  // rays_sampler_ = ray_sampler->GetInstance();
  // // select rays sample
  // if(ray_sampler_-> ray_sample_mode_ == 0){
  //   ray_sampler_ -> images_fnames_ = new_images_fnames;
  //   // std::cout << images_fnames[0] << std::endl;
  // }
  // // std::cout << ray_sampler_ -> ray_sample_mode_ <<std::endl;
  // else if (ray_sampler_->ray_sample_mode_ == 1){
  //   // generate all rays
  //   std::vector<Tensor> all_rays_o, all_rays_d, all_ranges;
  //   for (int i = 0; i < n_cameras_; ++i)
  //   { 
  //     AutoStudio::Camera camera = cameras[i];
  //     auto all_rays = camera.AllRaysGenerator();
  //     all_rays_o.push_back(all_rays.origins);
  //     all_rays_d.push_back(all_rays.dirs);
  //     all_ranges.push_back(all_rays.ranges);
  //   }
  //   Tensor all_rays_o_ = torch::stack(all_rays_o, 0).to(torch::kCUDA).contiguous();
  //   Tensor all_rays_d_ = torch::stack(all_rays_d, 0).to(torch::kCUDA).contiguous();
  //   Tensor all_ranges_ = torch::stack(all_ranges, 0).to(torch::kCUDA).contiguous();

  //   ray_sampler_->rays_ = {all_rays_o_, all_rays_d_, all_ranges_};
  //   std::cout << ray_sampler_->rays_.origins.sizes() <<std::endl;
  // }
  // else{

  // }
}

void Dataset::Normalize()
{ 
  auto poses = GetFullPose();
  const auto& config = global_data_->config_["dataset"];
  Tensor cam_pos = poses.index({Slc(), Slc(0, 3), 3}).clone();
  center_ = cam_pos.mean(0, false);
  Tensor bias = cam_pos - center_.unsqueeze(0);
  // std::cout << bias << std::endl;
  radius_ = torch::linalg_norm(bias, 2, -1, false).max().item<float>();
  // std::cout << radius_ << std::endl;
  cam_pos = (cam_pos - center_.unsqueeze(0)) / radius_;
  // std::cout << cam_pos.sizes() << std::endl;

  poses.index_put_({Slc(), Slc(0, 3), 3}, cam_pos);
  // std::cout << poses_.sizes() << std::endl;

  poses_ = poses.contiguous();
  c2w_ = poses_.clone();
  // std::cout << c2w_.sizes() << std::endl;
  w2c_ = torch::eye(4, CUDAFloat).unsqueeze(0).repeat({n_images_, 1, 1}).contiguous();
  w2c_.index_put_({Slc(), Slc(0, 3), Slc()}, c2w_.clone());
  w2c_ = torch::linalg_inv(w2c_);
  w2c_ = w2c_.index({Slc(), Slc(0, 3), Slc()}).contiguous();
  // std::cout << w2c_ << std::endl;
//   bounds_ = (bounds_ / radius_).contiguous();

//   Utils::TensorExportPCD(global_data_pool_->base_exp_dir_ + "/cam_pos.ply", poses_.index({Slc(), Slc(0, 3), 3}));
}



Tensor Dataset::GetFullPose()
{

  std::vector<Tensor> c2ws;
  for (int i = 0; i < n_camera_; ++i){
    auto camera = cameras_[i];
    auto images = camera.images_;
    for (int j = 0; j < camera.n_images_; ++j)
    {
      c2ws.push_back(images[j].c2w_);
    }
  }
  Tensor c2ws_tensor = torch::stack(c2ws, 0).reshape({-1, 4, 4});
  c2ws_tensor = c2ws_tensor.to(torch::kFloat32).to(torch::kCUDA).contiguous();
  return c2ws_tensor;
}


template <typename INPUT_T, typename OUTPUT_T>
std::vector<OUTPUT_T> Dataset::Convert2DVec1D(std::vector<std::vector<INPUT_T>> vec2d)
{
  std::vector<INPUT_T> vec1d;
  for(int i = 0; i < vec2d.size(); ++i){
    for (int j = 0; j < vec2d[0].size(); ++j)
    {
      vec1d.push_back(vec2d[i][j]);
    }
  }
  return vec1d;
}

void Dataset::Set_Shift(std::vector<int>& set, const int shift_const)
{
    for_each(set.begin(), set.end(), [&](int& elem){ elem = elem + shift_const;});
}

} // namespace AutoStudio