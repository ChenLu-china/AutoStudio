/**
* This file is part of auto_studio
* Copyright (C) 
**/


#define STB_IMAGE_IMPLEMENTATION
#include "../../utils/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../utils/stb_image_write.h"

#include <torch/torch.h>
#include "Image.h"
#include "../../Common.h"
#include "../../utils/cnpy.h"

namespace AutoStudio
{

using Tensor = torch::Tensor;
using namespace std;

Image::Image(const std::string& base_dir, const float& factor, const int& img_idx)
{ 
  std::string filename = std::string(8 - to_string(img_idx).length(), '0') + std::to_string(img_idx);
  img_fname_ = base_dir + "/" + "images" + "/" + filename + ".png";
  pose_fname_ = base_dir + "/" + "poses" + "/" + filename + ".npy";
  intri_fname_ = base_dir + "/"  + "intrinsic" + "/" + filename + ".npy";
  
  Tensor img_tensor = ReadImageTensor(img_fname_);
  if(img_tensor.sizes()[2] == 4){
    img_tensor_ = img_tensor.index({Slc(), Slc(), Slc(0, 3)});
  }
  else{
    img_tensor_ = img_tensor;
  }

  height_ = img_tensor_.size(0);
  width_ = img_tensor_.size(1);
  factor_ = factor_;

  // load c2w
  cnpy::NpyArray arr_pose = cnpy::npy_load(pose_fname_);
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  Tensor c2w = torch::from_blob(arr_pose.data<double>(), arr_pose.num_vals, options).to(torch::kFloat32).contiguous();
  Tensor c2w_4x4 = c2w.reshape({4, 4}).contiguous();
  Tensor c2w_3x4 = c2w_4x4.index({Slc(0, 3, 1), Slc()});
  c2w_ = c2w_3x4.reshape({3, 4}).contiguous();

  // load intrinsic
  cnpy::NpyArray arr_intri = cnpy::npy_load(intri_fname_);
  intri_ = torch::from_blob(arr_intri.data<double>(), arr_intri.num_vals, options).to(torch::kFloat32).contiguous();
  intri_ = intri_.reshape({3, 3}).contiguous();
  intri_.index_put_({Slc(0, 2), Slc(0, 3)}, intri_.index({Slc(0, 2), Slc(0, 3)}) / factor);

  // Todo: load this from input
  dist_param_ = torch::zeros({4}, options).to(torch::kFloat32);
}

std::tuple<Tensor, Tensor> Image::GenRaysTensor()
{
  auto option = torch::TensorOptions().dtype(torch::kInt32);
  int H = height_;
  int W = width_;
  
  Tensor ii = torch::linspace(0.f, H - 1.f, H, CUDAFloat);
  Tensor jj = torch::linspace(0.f, W - 1.f, W, CUDAFloat);
  auto ij = torch::meshgrid({ii, jj}, "ij");

  Tensor i = ij[0].reshape({-1});
  Tensor j = ij[1].reshape({-1});
  Tensor ij_ = torch::stack({i, j}, -1).to(torch::kCUDA).contiguous();
  auto [rays_o, rays_d] = Img2WorldRayFlex(ij_.to(torch::kInt32));
  return {rays_o, rays_d};
}

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

bool Image::WriteImageTensor(const std::string& path, Tensor img)
{
  Tensor out_img = (img * 255.f).clip(0.f, 255.f).to(torch::kUInt8).to(torch::kCPU).contiguous();
  stbi_write_png(path.c_str(), out_img.size(1), out_img.size(0), out_img.size(2), out_img.data_ptr(), 0);
  return true;
}

void Image::toCUDA()
{
  img_tensor_ = img_tensor_.to(torch::kCUDA).contiguous();
  c2w_ = c2w_.to(torch::kCUDA).contiguous();
  intri_ = intri_.to(torch::kCUDA).contiguous();
  dist_param_ = dist_param_.to(torch::kCUDA).contiguous();
}

void Image::toHost()
{
  img_tensor_ = img_tensor_.to(torch::kCPU).contiguous();
  c2w_ = c2w_.to(torch::kCPU).contiguous();
  intri_ = intri_.to(torch::kCPU).contiguous();
  dist_param_ = dist_param_.to(torch::kCPU).contiguous();
}

} // namespace AutoStudio