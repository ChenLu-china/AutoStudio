
#include <string>
#include <iostream>
#include <torch/torch.h>
#include <experimental/filesystem>

#include "Pipeline.h"
#include "../utils/GlobalData.h"

namespace fs = std::experimental::filesystem::v1;

namespace AutoStudio
{

using Tensor = torch::Tensor;

Runner::Runner(const std::string& conf_path)
{
  global_data_ = std::make_unique<GlobalData>(conf_path);
  // std::cout<< conf_path <<std::endl;
  const auto& config = global_data_->config_;
  task_name_ = config["task_name"].as<std::string>();
  base_dir_ = config["base_dir"].as<std::string>();

  base_exp_dir_ = config["base_exp_dir"].as<std::string>();
  global_data_->base_exp_dir_ = base_exp_dir_;

  fs::create_directories(base_exp_dir_);
  
  // Dataset 
  dataset_ = std::make_unique<Dataset>(global_data_.get());
  
}

} //namespace AutoStudio