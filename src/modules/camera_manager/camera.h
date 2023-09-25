/** @file   camera.h
 *  @author LuChen, 
 *  @brief 
*/

# pragma once

# include <string>
# include <torch/torch.h>

namespace AutoStudio{

using Tensor = torch::Tensor;

const int CAMERA_COUNT = 3;

namespace camera{

struct alignas(32) Rays{
    torch::Tensor origins;
    torch::Tensor dirs;
};

class Camera{

public:
  Camera(const std::string& base_dir, const std::string& cam_name);
  
  std::string base_dir_;
  std::string cam_name_;

  int n_images_ = 0;
  Tensor poses_, c2w_, intrinsics_;
  
  // Tensor c2w_train_, w2c_train_, intrinsic_train_;
  std::vector<int> train_set_, test_set_, val_set_, split_info_;
  
  int height_, width_;
  Tensor images_tensor;  
};

} // namespace camera
} // namespace AutoStudio
