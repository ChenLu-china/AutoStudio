/**
* This file is part of autostudio
* Copyright (C) 
**/


#ifndef TINYMLP_H
#define TINYMLP_H

#include <tiny-cuda-nn/cpp_api.h>

#include "FieldModel.h"


namespace AutoStudio
{
using Tensor = torch::Tensor;
using IValue = torch::IValue;
using namespace torch::autograd;

class TMLP : public FieldModel
{
public:
    TMLP(GlobalData* global_data, int d_in, int d_out, int d_hidden, int n_hidden_layers);
    void InitParams();
    int LoadStates(const std::vector<Tensor>& states, int idx) override;
    std::vector<Tensor> States() override;
    std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups() override;
    void Reset() override;

    int d_in_, d_out_, d_hidden_, n_hidden_layers_;
    std::unique_ptr<tcnn::cpp::Module> module_;
    Tensor params_;
    Tensor query_pts_, query_output_;
    
    tcnn::cpp::Context tmlp_ctx_;
    float loss_scale_ = 128.f;
}; 

class TMLPInfo : public torch::CustomClassHolder
{
public:
    TMLP* tmlp_ = nullptr;
};

class TMLPFunction : public Function<TMLPFunction>
{
public:
    static variable_list forward(AutogradContext *ctx,
                                Tensor input,
                                Tensor params,
                                IValue TMLPInfo);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

} // namespace AutoStudio
#endif // TINYMLP_H