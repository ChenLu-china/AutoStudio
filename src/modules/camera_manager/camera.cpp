/** @file   camera.cpp
 *  @author LuChen, 
 *  @brief 
*/

#include <torch/torch.h>
#include <fmt/core.h>
#include "camera.h"
#include "../../utils/Image.h"
#include "../../utils/cnpy.h"
#include "../../common.h"

// namespace fs = std::experimental::filesystem::v1;
#include <iostream>
using namespace std;


namespace AutoStudio{

using Tensor = torch::Tensor;

AutoStudio::camera::Camera::Camera(const std::string& dir, const std::string& name){
  base_dir_ = dir;
  cam_name_ = name;
    
  // Load images
  { 
    std::vector<Tensor> images;
    std::ifstream image_list(base_dir_ + "/" + cam_name_ + "/image_list.txt");
    std::string image_path;
    while (std::getline(image_list, image_path)){
      ++n_images_;
      images.push_back(AutoStudio::Image::get_instance().ReadImageTensor(image_path).to(torch::kCPU));
    }
    // std::cout<< images.size()<<std::endl;
    height_ = images[0].size(0);
    width_ = images[0].size(1);
    images_tensor = torch::stack(images, 0).contiguous();
  }
  
  // Load poses and intrisics
  {
    std::vector<Tensor> poses;
    std::vector<Tensor> intrinsics;

    for (int i = 0; i < n_images_; ++i){
      // load poses
      std::string filename = std::string(8 - to_string(i).length(), '0') + std::to_string(i);
      std::string pose_path = base_dir_ + "/" + cam_name_ + "/poses" + "/" + filename + ".npy";
      // std::cout << pose_path << std::endl;
      cnpy::NpyArray arr_pose = cnpy::npy_load(pose_path);
      auto options = torch::TensorOptions().dtype(torch::kFloat64);
      Tensor pose = torch::from_blob(arr_pose.data<double>(), arr_pose.num_vals, options).to(torch::kFloat32).to(torch::kCUDA);
      poses.push_back(pose);
      // if(i == 0){
      //   std::cout << pose << std::endl;      
      // }
      // load intrinsic
      std::string intrinsic_path = base_dir_ + "/" + cam_name_ + "/intrinsic" + "/" + filename + ".npy"; 
      cnpy::NpyArray arr_intri = cnpy::npy_load(intrinsic_path);
      Tensor intrinsic = torch::from_blob(arr_intri.data<double>(), arr_intri.num_vals, options).to(torch::kFloat32).to(torch::kCUDA);
      intrinsics.push_back(intrinsic);
    }
    poses_ = torch::stack(poses, 0).reshape({-1 ,4, 4}).contiguous();
    intrinsics_ = torch::stack(intrinsics, 0).reshape({-1, 3, 3}).contiguous();
    poses_ = poses_.to(torch::kCUDA).contiguous();
    intrinsics_ = intrinsics_.to(torch::kCUDA).contiguous();
  }

  // Load train/test/val split info
  try
  {
    cnpy::NpyArray split_arr = cnpy::npy_load(base_dir_ + "split.npy");
    CHECK_EQ(split_arr.shape[0], n_images_);
    
    auto sp_arr_ptr = split_arr.data<unsigned char>();
    for(int i = 0; i <n_images_; ++i)
    {
      int st = sp_arr_ptr[i];
      split_info_.push_back(st);
      if ((st & 1) == 1) train_set_.push_back(i);
      if ((st & 2) == 2) test_set_.push_back(i);
      if ((st & 4) == 4) val_set_.push_back(i);
    }
  }
  catch(...)
  {
    for (int i = 0; i < n_images_; i++) {
      if (i % 8 == 0) test_set_.push_back(i);
      else train_set_.push_back(i);
    }
  }

  std::cout << fmt::format("Amount of {}'s train/test/val: {}/{}/{}",
                          cam_name_, train_set_.size(), test_set_.size(), val_set_.size()) << std::endl;
  
  // std::cout << train_set << std::endl;
  // auto options = torch::TensorOptions().dtype(torch::kInt32);
  // train_set_ = torch::from_blob(train_set.data(), train_set.size(), options).contiguous();
  std::cout << test_set_ << std::endl;
  
}
} // namespace AutoStudio