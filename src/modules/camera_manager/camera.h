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
  int height_, width_;
  Tensor poses_, c2w_, intrinsics_;
  
  Tensor images_tensor;  
//   Tensor imgs = 
};

} // namespace camera
} // namespace AutoStudio
