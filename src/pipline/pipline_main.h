/** @file   pipline.h
 *  @author LuChen, 
 *  @brief 
*/

#pragma once

#include <string>
#include <memory>
#include <tuple>
#include <torch/torch.h>
#include "../utils/GlobalData.h"

namespace AutoStudio{
namespace run{

class Runner{
  using Tensor = torch::Tensor;

public:
  Runner(const std::string& config_path);

  void Train();
  void Render();

  // task information
  std::string task_name_, base_dir_, base_exp_dir_; 
  std::unique_ptr<GlobalData> global_data_;
};

} // namespace run
} //namespace AutoStudio