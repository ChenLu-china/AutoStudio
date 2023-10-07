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
    void set_shift(std::vector<int>& set, const int shift_const);

public:
    int n_images_ = 0;

    GlobalData* global_data_;
    Tensor center_;
    float radius_;
    
    Tensor train_set_, val_set_, test_set_;
    Tensor images_, poses_, c2w_, w2c_, intrinsics_;
    Tensor c2w_train_;
};

} // namespace AutoStudio

#endif // DATASET_H

