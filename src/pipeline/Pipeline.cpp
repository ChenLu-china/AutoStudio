/**
* This file is part of autostudio
* Copyright (C) 
**/


#include <string>
#include <iostream>
#include <torch/torch.h>
#include <experimental/filesystem>

#include "Pipeline.h"

#include "../utils/GlobalData.h"
#include "../utils/CustomOps/CustomOps.h"

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
  
  pts_batch_size_ = config["train"]["pts_batch_size"].as<int>();
  end_iter_ = config["train"]["end_iter"].as<int>();
  vis_freq_ = config["train"]["vis_freq"].as<int>();
  report_freq_ = config["train"]["report_freq"].as<int>();
  stats_freq_ = config["train"]["stats_freq"].as<int>();
  save_freq_ = config["train"]["save_freq"].as<int>();
  learning_rate_ = config["train"]["learning_rate"].as<float>();
  learning_rate_alpha_ = config["train"]["learning_rate_alpha"].as<float>();
  learning_rate_warm_up_end_iter_ = config["train"]["learning_rate_warm_up_end_iter"].as<int>();
  ray_march_init_fineness_ = config["train"]["ray_march_init_fineness"].as<float>();
  ray_march_fineness_decay_end_iter_ = config["train"]["ray_march_fineness_decay_end_iter"].as<int>();
  tv_loss_weight_ = config["train"]["tv_loss_weight"].as<float>();
  disp_loss_weight_ = config["train"]["disp_loss_weight"].as<float>();
  var_loss_weight_ = config["train"]["var_loss_weight"].as<float>();
  var_loss_start_ = config["train"]["var_loss_start"].as<int>();
  var_loss_end_ = config["train"]["var_loss_end"].as<int>();
  gradient_scaling_start_ = config["train"]["gradient_scaling_start"].as<int>();
  gradient_scaling_end_ = config["train"]["gradient_scaling_end"].as<int>();


  // Dataset 
  dataset_ = std::make_unique<Dataset>(global_data_.get());
  // dataset_->sampler_->TestRays();
  
  // Model
  model_pip_ = std::make_unique<ModelPipline>(global_data_.get(), dataset_->n_images_);
  
  //optimize
  optimizer_ = std::make_unique<torch::optim::Adam>(model_pip_->OptimParamGroups());
}


void Runner::UpdateAdaParams()
{
  // Update ray march fineness
  if (iter_step_ >= ray_march_fineness_decay_end_iter_) {
    global_data_->ray_march_fineness_ = 1.f;
  }
  else {
    float progress = float(iter_step_) / float(ray_march_fineness_decay_end_iter_);
    global_data_->ray_march_fineness_ = std::exp(std::log(1.f) * progress + std::log(ray_march_init_fineness_) * (1.f - progress));
  }
  // Update learning rate
  float lr_factor;
  if (iter_step_ >= learning_rate_warm_up_end_iter_) {
    float progress = float(iter_step_ - learning_rate_warm_up_end_iter_) /
                     float(end_iter_ - learning_rate_warm_up_end_iter_);
    lr_factor = (1.f - learning_rate_alpha_) * (std::cos(progress * float(M_PI)) * .5f + .5f) + learning_rate_alpha_;
  }
  else {
    lr_factor = float(iter_step_) / float(learning_rate_warm_up_end_iter_);
  }
  float lr = learning_rate_ * lr_factor;
  for (auto& g : optimizer_->param_groups()) {
    g.options().set_lr(lr);
  }
  // Update gradient scaling ratio
  {
    float progress = 1.f;
    if (iter_step_ < gradient_scaling_end_) {
      progress = std::max(0.f,
          (float(iter_step_) - gradient_scaling_start_) / (gradient_scaling_end_ - gradient_scaling_start_ + 1e-9f));
    }
    global_data_->gradient_scaling_progress_ = progress;
  }
}


void Runner::Train()
{
  global_data_->mode_ = RunningMode::TRAIN;

  std::string log_dir = base_exp_dir_ + "/logs";
  fs::create_directories(log_dir);

  std::vector<float> mse_records;
  float time_per_iter = 0.f;
  // StopWatch clock;
  
  float psnr_smooth = -1.0;
  UpdateAdaParams();

  {
    global_data_ -> iter_step_ = iter_step_;
    for (; iter_step_ < end_iter_;){
      global_data_->backward_nan_ = false;
      int cur_batch_size = int(pts_batch_size_ / global_data_->meaningful_sampled_pts_per_ray_) >> 4 << 4;
      std::cout << cur_batch_size << std::endl;
      dataset_->sampler_->batch_size_ = cur_batch_size;
      auto [train_rays, gt_colors, emb_idx] = dataset_->sampler_->GetTrainRays();
      std::cout << "train data size:" << train_rays.origins.sizes() << std::endl;
      std::cout << "train color size:" << gt_colors.sizes() << std::endl;
      // std::cout << "train emb_idx size:" << emb_idx << std::endl;

      Tensor& rays_o = train_rays.origins;
      Tensor& rays_d = train_rays.dirs;
      Tensor& ranges = train_rays.ranges;
      
      auto render_result = model_pip_->field_->Render(rays_o, rays_d, ranges, emb_idx);
      
      Tensor pred_colors = render_result.colors.index({Slc(0, cur_batch_size)});
      Tensor disparity = render_result.disparity;
      Tensor color_loss = torch::sqrt((gt_colors - pred_colors).square() + 1e-4f).mean();
      Tensor disparity_loss = disparity.square().mean();
      Tensor edge_feats = render_result.edge_feats;
      Tensor tv_loss = (edge_feats.index({Slc(), 0}) - edge_feats.index({Slc(), 1})).square().mean();
      
      Tensor sampled_weights = render_result.weights;
      Tensor idx_start_end = render_result.idx_start_end;
      Tensor sampled_var = CustomOps::WeightVar(sampled_var, idx_start_end);
      Tensor var_loss = (sampled_var + 1e-2).sqrt().mean();
      
      float var_loss_weight = 0.f;
      if (iter_step_ > var_loss_end_) {
        var_loss_weight = var_loss_weight_;
      }
      else if (iter_step_ > var_loss_start_) {
        var_loss_weight = float(iter_step_ - var_loss_start_) / float(var_loss_end_ - var_loss_start_) * var_loss_weight_;
      }
      
      Tensor loss = color_loss + var_loss * var_loss_weight + 
                    disparity_loss * disp_loss_weight_ + 
                    tv_loss * tv_loss_weight_;

      float mse = (pred_colors - gt_colors).square().mean().item<float>();
      float psnr = 20.f * std::log10(1 / std::sqrt(mse));
      psnr_smooth = psnr_smooth < 0.f ? psnr : psnr * .1f + psnr_smooth * .9f;
      CHECK(!std::isnan(pred_colors.mean().item<float>()));
      CHECK(!std::isnan(gt_colors.mean().item<float>()));
      CHECK(!std::isnan(mse));
      
      if (loss.requires_grad()){
        optimizer_->zero_grad();
        loss.backward();
        if (global_data_->backward_nan_){
          std::cout << "Nan!" << std::endl;
        }
        else{
          optimizer_->step();
        }
      }
      mse_records.push_back(mse);

      iter_step_++;
      global_data_->iter_step_ = iter_step_;

      if (iter_step_ % report_freq_ == 0) {
        std::cout << fmt::format(
            "Iter: {:>6d} PSNR: {:.2f} NRays: {:>5d} OctSamples: {:.1f} Samples: {:.1f} MeaningfulSamples: {:.1f} IPS: {:.1f} LR: {:.4f}",
            iter_step_,
            psnr_smooth,
            cur_batch_size,
            global_data_pool_->sampled_oct_per_ray_,
            global_data_pool_->sampled_pts_per_ray_,
            global_data_pool_->meaningful_sampled_pts_per_ray_,
            1.f / time_per_iter,
            optimizer_->param_groups()[0].options().get_lr())
                  << std::endl;
      }
      UpdateAdaParams();
    }
  }
}

void Runner::Execute()
{
  std::string mode = global_data_->config_["mode"].as<std::string>();
  std::cout << mode << std::endl;
  if (mode == "train"){
    Train();
  }
}



} //namespace AutoStudio