/**
* This file is part of auto_studio
* Copyright (C) 
* @file   camera.h
* @author LuChen, 
* @brief 
*/


#ifndef CAMERA_H
#define CAMERA_H
#include <string>
#include <torch/torch.h>
#include "Rays.h"
#include "Image.h"


namespace AutoStudio
{

using Tensor = torch::Tensor;

const int CAMERA_COUNT = 3;

class Camera
{
public:
  // functions
  Camera(const std::string& base_dir, const std::string& cam_name, float factor, std::vector<float> bounds_factor);

  // parameters
  std::string base_dir_;
  std::string cam_name_;

  int n_images_ = 0;
  float factor_;
  
  // Tensor c2w_train_, w2c_train_, intrinsic_train_;
  std::vector<AutoStudio::Image> images_;
  std::vector<int> img_idx_, train_set_, test_set_, val_set_, split_info_;
  std::vector<float> bounds_factor_;

  Tensor poses_, c2ws_, intrinsics_, dist_params_;
};

} // namespace AutoStudio

#endif // CAMERA_H