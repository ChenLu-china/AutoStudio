/** @file   pipline.h
 *  @author LuChen, 
 *  @brief 
*/


#ifndef PIPELINE_H
#define PIPELINE_H
#include <string>
#include <memory>
#include <tuple>
#include <torch/torch.h>
#include "ModelPipline.h"
#include "../utils/GlobalData.h"
#include "../dataset/Dataset.h"


namespace AutoStudio
{

class Runner
{

using Tensor = torch::Tensor;

public:
  Runner(const std::string& config_path);

  void Execute();

  void Train();
  void Render();
  void TestImages();
  void VisualizeImage(int idx);
  void UpdateAdaParams();
  void LoadCheckpoint(const std::string& path);
  void SaveCheckpoint();
  std::tuple<Tensor, Tensor, Tensor> RenderWholeImage(Tensor rays_o, Tensor rays_d, Tensor ranges);
  
  unsigned iter_step_ = 0;
  unsigned end_iter_;
  unsigned report_freq_, vis_freq_, stats_freq_, save_freq_;
  unsigned pts_batch_size_;

  float ray_march_init_fineness_;
  int ray_march_fineness_decay_end_iter_;
  int var_loss_start_, var_loss_end_;
  int gradient_scaling_start_, gradient_scaling_end_;
  float learning_rate_, learning_rate_alpha_, learning_rate_warm_up_end_iter_;
  float gradient_door_end_iter_;
  float var_loss_weight_, tv_loss_weight_, disp_loss_weight_;

  // task information
  std::string task_name_, base_dir_, base_exp_dir_; 
  std::unique_ptr<GlobalData> global_data_;
  std::unique_ptr<Dataset> dataset_;
  std::unique_ptr<ModelPipline> model_pip_;
  std::unique_ptr<torch::optim::Adam> optimizer_;
};

} //namespace AutoStudio

#endif // PIPELINE_H