/**
* This file is part of auto_studio
* Copyright (C) 
*  @file   camera.h
*  @author LuChen, 
*  @brief 
*/
#pragma once
#include <torch/torch.h>
#include "../../common/include/BaseModel.h"
#include "../../../utils/GlobalData.h"

namespace AutoStudio
{
using Tensor = torch::Tensor;

class FieldModel : public BaseModel
{
private:
    /* data */
public:
  virtual Tensor Query(const Tensor& coords) {
    CHECK(false) << "Not implemented";
    return Tensor();
  }

  virtual Tensor Query(const Tensor& coords, const Tensor& anchors) {
    CHECK(false) << "Not implemented";
    return Tensor();
  }

  virtual Tensor AnchoredQuery(const Tensor& coords, const Tensor& anchors) {
    CHECK(false) << "Not implemented";
    return Tensor();
  }

  GlobalData* global_data_;
};


} // namespace AutoStudio