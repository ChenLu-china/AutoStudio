/**
* This file is part of auto_studio
* Copyright (C) 
**/


#ifndef DATASET_H
#define DATASET_H
#include <torch/torch.h>


namespace Auto_Studio
{

using Tensor = torch::Tensor;

class GlobalData;

class Dataset
{
private:
    /* data */
public:
    Dataset(GlobalData *global_data);
    ~Dataset();

public:
    int n_images_ = 0;

    GlobalData* global_data_;


};

} // namespace Auto_Studio

#endif // DATASET_H

