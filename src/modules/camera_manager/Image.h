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

    Image(const std::string& base_dir, const std::string& img_path, const float& factor, const int& img_idx);
    std::tuple<Tensor, Tensor> GenRaysTensor();
    std::tuple<Tensor, Tensor> Img2WorldRayFlex(const Tensor& ij);

    Tensor ReadImageTensor(const std::string& image_path);
    bool WriteImageTensor(const std::string& image_path, Tensor img);
    
    void toHost();
    void toCUDA();

    int height_, width_;
    float near_, far_, factor_;
    std::string img_fname_, pose_fname_, intri_fname_;
    Tensor img_tensor_, c2w_, w2c_, intri_, dist_param_;
    
};

} // namespace AutoStudio

#endif // IMAGE_H