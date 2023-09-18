/**
* This file is part of auto_studio
* Copyright (C) 
**/


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <torch/torch.h>


namespace Auto_Studio
{

using Tensor = torch::Tensor;

Tensor Image::ReadImageTensor(const std::string& path)
{
  int w, h, n;
  unsigned char *idata = stbi_load(path.c_str(), &w, &h, &n, 0);

  Tensor img = torch::empty({ h, w, n }, CPUUInt8);
  std::memcpy(img.data_ptr(), idata, w * h * n);
  stbi_image_free(idata);

  img = img.to(torch::kFloat32).to(torch::kCPU) / 255.f;
  return img;
}

} // namespace Auto_Studio