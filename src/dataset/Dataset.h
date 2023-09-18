/**
* This file is part of auto_studio
* Copyright (C) 
**/


#ifndef DATASET_H
#define DATASET_H
#include <torch/torch.h>

using Tensor = torch::Tensor;


namespace Auto_Studio
{

class GlobalData;

class Dataset
{
private:
    /* data */
public:
    Dataset(GlobalData *global_data);
    ~Dataset();

protected:
    GlobalData* global_data_;


};

} // namespace Auto_Studio

#endif // DATASET_H

