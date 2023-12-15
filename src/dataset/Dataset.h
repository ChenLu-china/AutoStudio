/**
* This file is part of autostudio
* Copyright (C) 
* @file   
* @author 
* @brief 
*/


#ifndef DATASET_H
#define DATASET_H
#include <torch/torch.h>
#include "../Common.h"
#include "../utils/GlobalData.h"
#include "../modules/camera_manager/Camera.h"
#include "../modules/camera_manager/Sampler.h"


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
    enum class DataCode{C2W, W2C, INTRI, BOUND, NONE};
    
    void Normalize(); // deep process Image c2w
    void UpdateNormProc();
    void Set_Shift(std::vector<int>& set, const int shift_const);
    
    Tensor GetFullC2W_Tensor(bool device);
    Tensor GetFullIntri_Tensor(bool device);
    Tensor GetFullW2C_Tensor(bool device);
    Tensor GetFullData_Tensor(std::string dType, bool device);
    Tensor GetTrainData_Tensor(std::string dType, bool device);

    // tuple<Tensor, Tensor> GenRaysFlex(Tensor c2ws, Tensor w2cs, Tensor intris, int )
    // Tensor GetTrainC2W_Tensor(bool device);
    // Tensor GetTrainIntri_Tensor(bool device);
    // Tensor GetTrainW2C_Tensor(bool device);
    // Tensor GetTrainBound_Tensor(bool device);

    template <typename T>
    std::vector<T> GetFullC2W(bool device);

    template <typename T>
    std::vector<T> GetFullIntri(bool device);

    template <typename INPUT_T, typename OUTPUT_T>
    std::vector<OUTPUT_T> GetFullImage();

    template <typename T>
    std::vector<T> Flatten2DVector(const std::vector<std::vector<T>>& vec2d);
    
    DataCode DataShit(std::string dType);

public:
    std::string set_name_, set_sequnceid_;
    int n_images_ = 0;
    int n_camera_ = 0;
    
    GlobalData* global_data_;
    Sampler*  sampler_;

    Tensor center_;
    float radius_;
    Tensor c2w_, w2c_, poses_, intris_;
    Tensor train_set_, val_set_, test_set_;
    // Tensor images_, poses_, c2w_, w2c_, intrinsics_;
    Tensor c2w_train_;

    std::vector<Camera> cameras_;
    // std::unique_ptr<AutoStudio::RaySampler> ray_sampler_;
};

} // namespace AutoStudio

#endif // DATASET_H