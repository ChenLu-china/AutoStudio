/**
* This file is part of autostudio
* Copyright (C) 
**/


#ifndef DATASET_H
#define DATASET_H
#include <torch/torch.h>
#include "../utils/GlobalData.h"

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
    void Normalize();
public:
    int n_images_ = 0;

    GlobalData* global_data_;

    Tensor images_, poses_, intrinsics_;
};

} // namespace AutoStudio

#endif // DATASET_H

