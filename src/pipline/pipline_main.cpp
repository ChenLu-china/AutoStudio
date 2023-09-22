
#include <string>
#include <iostream>
#include <torch/torch.h>
#include <experimental/filesystem>

#include "pipline_main.h"
#include "../utils/GlobalData.h"

namespace fs = std::experimental::filesystem::v1;

namespace AutoStudio{

  using Tensor = torch::Tensor;

AutoStudio::run::Runner::Runner(const std::string& conf_path){
  global_data_ = std::make_unique<GlobalData>(conf_path);
  std::cout<< conf_path <<std::endl;
  const auto& config = global_data_->config_;
  task_name_ = config["task_name"].as<std::string>();
  base_dir_ = config["base_dir"].as<std::string>();

  const auto& cam_list = config["cam_list"];
  std::cout<< cam_list[0] <<std::endl;

}

} //namespace AutoStudio 