// /**
// * This file is part of autostudio
// * Copyright (C) 
// **/


// #ifndef RAYS_H
// #define RAYS_H
// #include <string>
// #include <torch/torch.h>
// #include "../../utils/GlobalData.h"

// namespace AutoStudio
// {

// using Tensor = torch::Tensor;

// struct alignas(32) Rays{
//     Tensor origins;
//     Tensor dirs;
// };

// struct alignas(32) RangeRays{
//     Tensor origins;
//     Tensor dirs;
//     Tensor ranges;
// };

// class RaySampler
// {

// public:
//     RaySampler(GlobalData* global_data_pool);
//     Tensor RandIdx();
//     RangeRays Train_Rays(); // out for modules
    
//     enum RaySampleMode {
//         SINGLE_IMAGE,
//         ALL_IMAGES,
//     };

// public:
//     RangeRays rays_;
//     RaySampleMode ray_sample_mode_;
//     std::vector<std::string> images_fnames_;  
//     Tensor c2w_train_, w2c_train_, intri_train_, H_train_, W_train_, bound_train_;
// };

// } // namespace AutoStudio

// #endif // RAYS_H