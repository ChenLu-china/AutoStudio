
#include <string>
#include <torch/torch.h>

#include "pipline_main.h"
#include "../utils/GlobalData.h"



namespace AutoStudio{
  using Tensor = torch::Tensor;

AutoStudio::run::Runner::Runner(const std::string& conf_path){
  global_data_pool_ = std::make_unique<GlobalData>(conf_path);
}

} //namespace AutoStudio 