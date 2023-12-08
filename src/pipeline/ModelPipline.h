/**
* This file is part of autostudio
* Copyright (C) 
**/


#ifndef MODELPIPELINE_H
#define MODELPIPELINE_H
#include <torch/torch.h>
#include "../utils/GlobalData.h"

#include "../modules/common/include/BaseModel.h"
#include "../modules/fields/include/FieldsFactory.h"


namespace AutoStudio
{

using Tensor = torch::Tensor;


class ModelPipline : public BaseModel
{
    
public:
    ModelPipline(GlobalData* global_data, int n_images);
    std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups() override;
    std::unique_ptr<FieldModel> field_;
    GlobalData* global_data_;

};

} // namespace AutoStudio

#endif // MODELPIPELINE_H