/**
* This file is part of auto_studio
* Copyright (C) 
*  @file   
*  @author LuChen, 
*  @brief 
*/

#include "include/TinyMLP.h"

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/torch.h>
#pragma warning(pop)
#else
#include <torch/torch.h>
#endif

#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>

#ifdef snprintf
#undef snprintf
#endif

#include <json/json.hpp>
#include <tiny-cuda-nn/cpp_api.h>
#include <iostream>

#include "../../Common.h"

#define CHECK_TS(x) CHECK(x.device().is_cuda()); CHECK(x.is_contiguous())

namespace AutoStudio
{

using Tensor = torch::Tensor;
using namespace torch::autograd;

void* void_data_ptr(torch::Tensor& tensor)
{
    switch (tensor.scalar_type())
    {
        case torch::kFloat32: return tensor.data_ptr<float>();
        case torch::kHalf: return tensor.data_ptr<torch::Half>();
        default: throw std::runtime_error{"Unknown precision torch->void"};
    }
}

c10::ScalarType torch_type(tcnn::cpp::Precision precision)
{
    switch (precision){
        case tcnn::cpp::Precision::Fp32: return torch::kFloat32;
        case tcnn::cpp::Precision::Fp16: return torch::kHalf;
        default: throw std::runtime_error{"Unknown precision tcnn->torch"};
    }
}

TORCH_LIBRARY(tcnn_wp, m)
{
  std::cout << "register TMLPInfo" << std::endl;
  m.class_<TMLPInfo>("TMLPInfo").def(torch::init());
}

TMLP::TMLP(GlobalData* global_data, int d_in, int d_out, int d_hidden, int n_hidden_layers)
{
    global_data_ = global_data;
    d_in_ = d_in;
    d_out_ = d_out;
    d_hidden_ = d_hidden;
    n_hidden_layers_ = n_hidden_layers;

    nlohmann::json config = {
        {"otype", "FullyFusedMLP"},
        {"activation", "ReLU"},
        {"output_activation", "None"},
        {"n_neurons", d_hidden},
        {"n_hidden_layers", n_hidden_layers}
    };

    module_ = std::unique_ptr<tcnn::cpp::Module>(tcnn::cpp::create_network(d_in_, d_out_, config));
    Tensor params = torch::zeros({int(module_->n_params())}, CUDAFloat);
    size_t seed = 19970826;
    module_->initialize_params(seed, params.data_ptr<float>());
    params_ = params.to(torch::kFloat32);
    params.requires_grad_(true);
}

variable_list TMLPFunction::forward(AutogradContext *ctx,
                                    Tensor input,
                                    Tensor params,
                                    IValue tmlp_info)
{
    ctx->set_materialize_grads(false);
    auto info_ptr = tmlp_info.toCustomClass<TMLPInfo>();
    ctx->saved_data["tmlp_info"] = tmlp_info;
    auto tmlp_wp = info_ptr->tmlp_;

    CHECK_TS(input);
    CHECK_TS(params);

}

variable_list TMLPFunction::backward(AutogradContext *ctx,
                                    variable_list grad_output)
{
    
}

} // namespace AutoStudio