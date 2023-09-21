/** @file   pipline.h
 *  @author LuChen, 
 *  @brief 
*/

#pragma once

#include <string>
#include <memory>
#include <tuple>
#include <torch/torch.h>


namespace AutoStudio{
namespace run{

class Runner{
  using Tensor = torch::Tensor;

public:
  Runner(const std::string& config_path);

  void Train();
  void Render();
  
};

} // namespace run
} //namespace AutoStudio