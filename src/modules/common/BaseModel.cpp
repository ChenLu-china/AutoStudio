/**
* This file is part of autostudio
* Copyright (C) 
**/

#include <torch/torch.h>
#include "./include/BaseModel.h"


namespace AutoStudio{

using Tensor = torch::Tensor;

int BaseModel::LoadStates(const std::vector<Tensor>& states, int idx){
    for (auto model : sub_models_){
        idx = model->LoadStates(states, idx);
    }
    return idx;
}

std::vector<Tensor> BaseModel::States() {
  std::vector<Tensor> ret;
  for (auto model : sub_models_) {
    auto cur_states = model->States();
    ret.insert(ret.end(), cur_states.begin(), cur_states.end());
  }
  return ret;
}

std::vector<torch::optim::OptimizerParamGroup> BaseModel::OptimParamGroups() {
  std::vector<torch::optim::OptimizerParamGroup> ret;
  for (auto model : sub_models_) {
    auto cur_params = model->OptimParamGroups();
    for (const auto& para_group : cur_params) {
      ret.emplace_back(para_group);
    }
  }
  return ret;
}

void BaseModel::RegisterSubPipe(BaseModel* sub_model) {
  sub_models_.push_back(sub_model);
}

void BaseModel::Reset() {
  for (auto model : sub_models_) {
    model->Reset();
  }
}

} // AutoStudio