/**
* This file is part of autostudio
* Copyright (C) 
**/


#ifndef DATASET_H
#define DATASET_H
#include <torch/torch.h>
#include "../utils/GlobalData.h"
#include "../modules/camera_manager/camera.h"
#include "../modules/camera_manager/rays.h"

namespace AutoStudio
{

using Tensor = torch::Tensor;

// class GlobalData;

class Dataset
{
private:
    /* data */
public:
    Dataset(GlobalData *global_data);
    
    void Normalize(); // deep process Image c2w
    void Set_Shift(std::vector<int>& set, const int shift_const);
    Tensor GetFullPose();

    template <typename INPUT_T, typename OUTPUT_T>
    std::vector<OUTPUT_T> GetFullImage();


    template <typename INPUT_T, typename OUTPUT_T>
    std::vector<OUTPUT_T> Convert2DVec1D(std::vector<std::vector<INPUT_T>> vec2d);

public:
    int n_images_ = 0;
    int n_camera_ = 0;

    GlobalData* global_data_;
    
    Tensor center_;
    float radius_;

    Tensor train_set_, val_set_, test_set_;
    Tensor images_, poses_, c2w_, w2c_, intrinsics_;
    Tensor c2w_train_;

    std::vector<Camera> cameras_;
    std::unique_ptr<AutoStudio::RaySampler> ray_sampler_;
};

} // namespace AutoStudio

#endif // DATASET_H

