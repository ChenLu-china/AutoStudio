/**
* This file is part of autostudio
* Copyright (C) 
**/

#include "include/HashMap.h"
#include "../../Common.h"
#include <Eigen/Eigen>
#include <torch/torch.h>

namespace AutoStudio{

using Tensor = torch::Tensor;
using namespace torch::autograd;

template<typename T>
__global__ void Hash3DVertexForwardKernel(int n_points, int n_volume,
                                         T * feat_pool, int* prim_pool, int* feat_local_idx, int* feat_local_size,
                                         Wec3f* bias_pool, 
                                         Wec3f* points_ptr,
                                         int* volume_idx,
                                         T* out_feat)
{
    int pts_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int level_idx = blockIdx.y;
    if (pts_idx >= n_points){
        return;
    }

    points_ptr = points_ptr + pts_idx;
    volume_idx = volume_idx + pts_idx;
    out_feat = out_feat + pts_idx * (N_LEVELS * N_CHANNELS);
    
    Wec3f pt = points_ptr[0];

}

variable_list Hash3DVertexFunction::forward(AutogradContext *ctx,
                              Tensor feat_pool,
                              torch::IValue hash3d_info)
{
    auto info_ptr = hash3d_info.toCustomClass<Hash3DVertexInfo>();
    ctx->saved_data["hash3d_info"] = hash3d_info;
    Tensor& points = info_ptr->hash3dvertex_->query_points_;
    Tensor& volume_idx = info_ptr->hash3dvertex_->query_volume_idx_;
    Tensor& prim_pool = info_ptr->hash3dvertex_->prim_pool_;
    Tensor& bias_pool = info_ptr->hash3dvertex_->bias_pool_;
    Tensor& feat_local_idx = info_ptr->hash3dvertex_->feat_local_idx_;
    Tensor& feat_local_size = info_ptr->hash3dvertex_->feat_local_size_;
    CHECK(points.device().is_cuda());
    CHECK(volume_idx.device().is_cuda());

    int n_points = points.size(0);
    int n_volumes = info_ptr->hash3dvertex_->n_volumes_;

    const unsigned thread_cap = 512;
    dim3 block_dim = {unsigned(thread_cap), 1, 1};
    dim3 grid_dim = {DivUp(n_points, thread_cap), unsigned(N_LEVELS), 1};

    Tensor out_feat = torch::zeros({n_points, N_LEVELS * N_CHANNELS }, CUDAFloat);
    CHECK(out_feat.is_contiguous());
    
    Tensor feat_pool_true = feat_pool.to(torch::kFloat16).contiguous();
    return { out_feat.to(torch::kFloat32) };
}

variable_list Hash3DVertexFunction::backward(AutogradContext *ctx, variable_list grad_output)
{
    auto info_ptr = ctx->saved_data["hash3d_info"].toCustomClass<Hash3DVertexInfo>();
    Tensor& points = info_ptr->hash3dvertex_->query_points_;
    
    int pool_size = info_ptr->hash3dvertex_->pool_size_;
    Tensor true_grad_out = torch::zeros({pool_size, N_CHANNELS}, CUDAFloat);
    return { true_grad_out.to(torch::kFloat32), Tensor() };
}

} // namespce AutoStudio