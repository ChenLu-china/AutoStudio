/**
* This file is part of autostudio
* Copyright (C) 
**/
#include "SHShader.h"

namespace AutoStudio
{

SHShader::SHShader(GlobalData* global_data)
{   

    std::cout << "SHSader::SHShade" << std::endl;
    global_data_ = global_data;
    auto config =  global_data->config_["models"]["shader"];
    d_in_ = config["d_in"].as<int>();
    d_out_ = config["d_out"].as<int>();
    degree_ = config["degree"].as<int>();
    d_hidden_ = config["d_hidden"].as<int>();
    n_hidden_ = config["n_hidden"].as<int>();

    // MLP
    mlp_  = std::make_unique<TMLP>(global_data_, d_in_, d_out_, d_hidden_, n_hidden_);
}

Tensor SHShader::Query(const Tensor& feats, const Tensor& dirs)
{   
    Tensor enc = SHEncode(dirs);
    Tensor input = torch::cat({feats, enc}, -1);
    Tensor output = mlp_->Query(input);
    float eps = 1e-3f;
    return (1.f + 2.f * eps) / (1.f + torch::exp(-output)) - eps;
} 


std::vector<torch::optim::OptimizerParamGroup> SHShader::OptimParamGroups()
{    
    float lr = global_data_->learning_rate_;
    auto opt = std::make_unique<torch::optim::AdamOptions>(lr);
    opt->betas() = {0.9, 0.99};
    opt->eps() = 1e-15;
    opt->weight_decay() = 1e-6;

    std::vector<Tensor> params;
    params.push_back(mlp_->params_);
    return {torch::optim::OptimizerParamGroup(params, std::move(opt))};
    
}

int SHShader::LoadStates(const std::vector<Tensor> &states, int idx)
{
    mlp_->params_.data().copy_(states[idx++]);
    return idx;
}

std::vector<Tensor> SHShader::States()
{
    std::vector<Tensor> ret;
    ret.push_back(mlp_->params_.data());
    return ret;
}

void SHShader::Reset() {
  mlp_->InitParams();
}

} // namespace AutoStudio