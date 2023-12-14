/**
* This file is part of autostudio
* Copyright (C) 
* @file   
* @author 
* @brief 
*/


#ifndef OCTREEMAP_H
#define OCTREEMAP_H
#include <torch/torch.h>
#include <Eigen/Eigen>
#include "Octree.h"
#include "../include/HashMap.h"
#include "../../common/shader/SHShader.h"
#include "../../common/include/FieldModel.h"
#include "../../../utils/CustomOps/CustomOps.h"
#include "../../../utils/CustomOps/FlexOps.h"
#include "../../../utils/CustomOps/Scatter.h"


namespace AutoStudio
{

using Tensor = torch::Tensor;


class OctreeMapInfo : public torch::CustomClassHolder {
public:
  SampleResultFlex* sample_result;
};

class OctreeMap : public FieldModel
{

public:
    OctreeMap(GlobalData* global_data);

    using FieldModel::GetSamples;
    SampleResultFlex GetSamples(const Tensor& rays_o, 
                                const Tensor& rays_d, 
                                const Tensor& bounds) override;    
    std::tuple<Tensor, Tensor> GetEdgeSamples(int n_pts);
    
    
    void UpdateOctNodes(const SampleResultFlex& sample_result,
                        const Tensor& sampled_weights,
                        const Tensor& sampled_alpha) override;

    std::vector<Tensor> States() override;
    int LoadStates(const std::vector<Tensor>& states, int idx) override;
    std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups() override;

    RenderResult Render(const Tensor& rays_o, 
                        const Tensor& rays_d, 
                        const Tensor& ranges, 
                        const Tensor& emb_idx) override;

    void VisOctree();
    
    std::unique_ptr<Octree> octree_;
    std::unique_ptr<Hash3DVertex> hashmap_;
    std::unique_ptr<SHShader> shader_;
    std::vector<int> sub_div_milestones_;
    
    int compact_freq_;
    int max_oct_intersect_per_ray_; 
    float global_near_;
    float sample_l_;
    bool scale_by_dis_;
    bool use_app_emb_;
    Tensor app_emb_;
    SampleResultFlex sample_result_;
    BGColorType bg_color_type_ = BGColorType::rand_noise;
};

torch::Tensor FilterIdxBounds(const torch::Tensor& idx_bounds,
                              const torch::Tensor& mask);

} // namespace AutoStudio

#endif // OCTREEMAP_H