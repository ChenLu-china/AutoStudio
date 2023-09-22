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
  Camera(const std::string& cam_name);
  
  std::string cam_name;
//   Tensor imgs = 
};

} // namespace camera
} // namespace AutoStudio
