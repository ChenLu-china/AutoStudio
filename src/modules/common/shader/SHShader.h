/**
* This file is part of autostudio
* Copyright (C) 
**/

#ifndef SHSHADER_H
#define SHSHADER_H 

#include <torch/torch.h>
#include "../mlp/TinyMLP.h"
#include "../include/SHModel.h"
#include "../../../utils/GlobalData.h"

namespace AutoStudio
{
using Tensor = torch::Tensor;

class SHShader : public Shader
{
private:
    /* data */
public:
    SHShader(GlobalData* gloable_data);

    Tensor Query(const Tensor& feats, const Tensor& dirs) override;
    Tensor SHEncode(const Tensor& dirs);
    std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups() override;
    int LoadStates(const std::vector<Tensor> &states, const int idx ) override;
    std::vector<Tensor> States() override;
    void Reset() override;

    int d_in_, d_out_, degree_, d_hidden_, n_hidden_;
    std::unique_ptr<TMLP> mlp_;
};


} // namespace AutoStudio
#endif // SHSHADER_H