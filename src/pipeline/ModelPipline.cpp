/**
* This file is part of autostudio
* Copyright (C) 
**/

#include "ModelPipline.h"

namespace AutoStudio
{

using Tensor = torch::Tensor;

ModelPipline::ModelPipline(GlobalData* global_data)
{   
    std::cout << "Doing ModelPipline" << std::endl;
    auto field_factor = FieldsFactory(global_data);
    
    // field construction
    field_ = field_factor.CreateField();
    RegisterSubPipe(field_.get());

    std::cout << "sub_models size is " << sub_models_.size() << std::endl;
    // sh construction 
}

RenderResult ModelPipline::Render(const Tensor& rays_o, const Tensor& rays_d, const Tensor& ranges, const Tensor& emb_idx)
{
#ifdef PROFILE
#endif
    int n_rays = rays_o.sizes()[0];
    std:: cout << n_rays << std::endl;
    sample_result_ = field_->GetSamples(rays_o, rays_d, ranges);
    std::cout << sample_result_.pts.sizes() << std::endl;

    return { Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor() };
}


std::vector<torch::optim::OptimizerParamGroup> ModelPipline::OptimParamGroups()
{
    std::vector<torch::optim::OptimizerParamGroup> ret;
    std::cout << "pass0.1" << std::endl;
    for (auto model : sub_models_){
        auto cur_params = model->OptimParamGroups();
        for (const auto& param_group : cur_params){
            ret.emplace_back(param_group);
        }
    }
    
    {
        auto opt = std::make_unique<torch::optim::AdamOptions>(global_data_->learning_rate_);
        opt->betas() = {0.9, 0.99};
        opt->eps() = 1e-15;
        opt->weight_decay() = 1e-6;

        std::vector<Tensor> params;
        // params.push_back(app);
        ret.emplace_back(std::move(params), std::move(opt));
    }
    std::cout << "pass0.2" << std::endl;
    return ret;
}

} // namespace AutoStudio   