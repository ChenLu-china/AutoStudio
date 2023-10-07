/** @file   camera.h
 *  @author LuChen, 
 *  @brief 
*/

# pragma once

#include <string>
#include <torch/torch.h>
#include "rays.h"

namespace AutoStudio{

using Tensor = torch::Tensor;

const int CAMERA_COUNT = 3;

class Camera{

public:
  Camera(const std::string& base_dir, const std::string& cam_name, float factor);
  
  RangeRays AllRaysGenerator();

  std::tuple<Tensor, Tensor> Img2WorldRayFlex(const Tensor& img_indices, const Tensor& ij);
  Tensor CameraUndistort(const Tensor& cam_xy, const Tensor& dist_params);

  std::string base_dir_;
  std::string cam_name_;

  int n_images_ = 0;
  int height_, width_;
  float factor_;
  
  // Tensor c2w_train_, w2c_train_, intrinsic_train_;
  std::vector<int> img_idx_, train_set_, test_set_, val_set_, split_info_;
  
  AutoStudio::RangeRays rays;
 
  Tensor poses_, c2w_, intrinsics_, dist_params_;
  Tensor images_tensor_;  
};

} // namespace AutoStudio
