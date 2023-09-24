/**
* This file is part of auto_studio
* Copyright (C) 
**/


#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <torch/torch.h>


namespace AutoStudio
{

using Tensor = torch::Tensor;

class Image
{
public:

    static inline Image& get_instance()
    {
        static Image instance;
        return instance;
    }

    Tensor ReadImageTensor(const std::string& image_path);
    bool WriteImageTensor(const std::string& image_path, Tensor img);
    
};

} // namespace AutoStudio

#endif // IMAGE_H