/**
* This file is part of autostudio
* Copyright (C) 
**/

#pragma once
#include <torch/torch.h>

namespace AutoStudio{

using Tensor = torch::Tensor;

class BaseModel{

public:
    virtual int LoadStates(const std::vector<Tensor>& states, int idx);
    virtual std::vector<Tensor> States();
    virtual std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups();
    virtual void Reset();
    void RegisterSubPipe(BaseModel* sub_model);

    std::vector<BaseModel*> sub_models_;
};

} // AutoStudio
