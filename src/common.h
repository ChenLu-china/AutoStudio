#pragma once 

#define CUDAFloat torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
#define CPUUInt8 torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)
#define Slc torch::indexing::Slice