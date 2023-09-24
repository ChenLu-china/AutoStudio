/** @file   camera.cpp
 *  @author LuChen, 
 *  @brief 
*/

#include <torch/torch.h>
#include "camera.h"
#include "../../utils/Image.h"
#include "../../utils/cnpy.h"
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
      // for (int i = 0; i < n_images_; i++) {
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
    std::cout << poses_.sizes() << std::endl;
    intrinsics_ = torch::stack(intrinsics, 0).reshape({-1, 3, 3}).contiguous();
    std::cout << intrinsics_.sizes() << std::endl;
    poses_ = poses_.to(torch::kCUDA).contiguous();
    intrinsics_ = intrinsics_.to(torch::kCUDA).contiguous();
  }
  
  // n_images_ = images.size(0);
}
} // namespace AutoStudio