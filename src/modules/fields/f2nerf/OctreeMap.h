/**
* This file is part of autostudio
* Copyright (C) 
**/


#ifndef OCTREEMAP_H
#define OCTREEMAP_H
#include <torch/torch.h>
#include "Octree.h"
#include "../include/FieldModel.h"


namespace AutoStudio
{

using Tensor = torch::Tensor;

class OctreeMap : public FieldModel
{

public:
    OctreeMap(GlobalData* global_data);
    SampleResultFlex GetSamples(const Tensor& rays_o, const Tensor& rays_d, const Tensor& bounds) override;

    void UpdateOctNodes(const SampleResultFlex& sample_result,
                    const Tensor& sampled_weights,
                    const Tensor& sampled_alpha) override;

    std::vector<Tensor> States() override;
    int LoadStates(const std::vector<Tensor>& states, int idx) override;

    std::unique_ptr<Octree> octree_;
    std::vector<int> sub_div_milestones_;
    int compact_freq_;
    int max_oct_intersect_per_ray_;
    float global_near_;
    float sample_l_;
    bool scale_by_dis_;
};

} // namespace AutoStudio

#endif // OCTREEMAP_H