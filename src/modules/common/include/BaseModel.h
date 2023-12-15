/**
* This file is part of autostudio
* Copyright (C) 
* @file   camera.h
* @author LuChen, 
* @brief 
*/


#ifndef BASEMODEL_H
#define BASEMODEL_H
#include <torch/torch.h>
#include <vector>


namespace AutoStudio
{

using Tensor = torch::Tensor;

class BaseModel
{
public:
    virtual int LoadStates(const std::vector<Tensor>& states, int idx);
    virtual std::vector<Tensor> States();
    virtual std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups();
    virtual void Reset();
    void RegisterSubPipe(BaseModel* sub_model);

    std::vector<BaseModel*> sub_models_;
};

} // namespace AutoStudio

#endif // BASEMODEL_H