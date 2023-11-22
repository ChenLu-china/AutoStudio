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

struct RenderResult
{
    Tensor colors;
    Tensor first_oct_dis;
    Tensor disparity;
    Tensor edge_feats;
    Tensor depth;
    Tensor weights;
    Tensor idx_start_end;
};


class ModelPipline : public BaseModel
{
private:
    /* data */
public:
    ModelPipline(GlobalData* global_data);
    std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups() override;
    std::unique_ptr<FieldModel> field_;

    RenderResult Render(const Tensor& rays_o, const Tensor& rays_d, const Tensor& ranges, const Tensor& emb_idx);
        
    GlobalData* global_data_;
    SampleResultFlex sample_result_;
};

} // namespace AutoStudio

#endif // MODELPIPELINE_H