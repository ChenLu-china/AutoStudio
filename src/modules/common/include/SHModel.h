/**
* This file is part of autostudio
* Copyright (C) 
* @file   
* @author 
* @brief 
*/


#ifndef SHMODEL_H
#define SHMODEL_H
#include <torch/torch.h>
#include "BaseModel.h"
#include "../../../utils/GlobalData.h"


namespace AutoStudio
{

using Tensor = torch::Tensor;

class Shader : public BaseModel
{
public:
  virtual Tensor Query(const Tensor& feats, const Tensor& dirs) {
    CHECK(false) << "Not implemented";
    return Tensor();
  }
  GlobalData* global_data_;

  int d_in_, d_out_;
};

} // namespace AutoStudio

#endif // SHMODEL_H