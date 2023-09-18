/**
* This file is part of auto_studio
* Copyright (C) 
**/


#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <torch/torch.h>


namespace Auto_Studio
{

using Tensor = torch::Tensor;

class Image
{
public:
    Tensor ReadImageTensor(const std::string& image_path);
};

} // namespace Auto_Studio

#endif // IMAGE_H