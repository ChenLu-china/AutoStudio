/**
* This file is part of auto_studio
* Copyright (C) 
*  @file   
*  @author LuChen, 
*  @brief 
*/

#include "TinyMLP.h"

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

#include "../../../Common.h"

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

TORCH_LIBRARY(tmlp_wp, m)
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
    std::cout << params.sizes() << std::endl;
}

Tensor TMLP::Query(const Tensor& pts)
{   
    auto info = torch::make_intrusive<TMLPInfo>();
    
    int batch_size = pts.size(0);
    int batch_size_al = (batch_size + 255) / 256 * 256;
    auto pad_opt = torch::nn::functional::PadFuncOptions({0LL, 0LL, 0LL, (long long) (batch_size_al - batch_size)});
    Tensor input = torch::nn::functional::pad(pts, pad_opt);
    tmlp_ctx_.ctx.reset();
    info->tmlp_ = this;
    Tensor feat = TMLPFunction::apply(input, params_.to(torch::kFloat16), torch::IValue(info))[0];
    return feat.index({Slc(0, batch_size), Slc(0, d_out_)}).to(torch::kFloat32).contiguous();
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

    CHECK_EQ(input.scalar_type(), torch::kFloat32);
    CHECK_EQ(params.scalar_type(), torch_type(tmlp_wp->module_->param_precision()));

    CHECK_EQ(input.size(1), tmlp_wp->module_->n_input_dims());
    CHECK_EQ(params.size(0), tmlp_wp->module_->n_params());

    at::Device device = input.device();
    CHECK_EQ(input.device(), params.device());

    const at::cuda::CUDAGuard device_guard(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    uint32_t batch_size = input.size(0);
    // std::cout << "forward batch size is: " << batch_size << std::endl;
    torch::Tensor output = torch::rand({ batch_size, tmlp_wp->module_->n_output_dims() }, torch::TensorOptions().dtype(
        torch_type(tmlp_wp->module_->output_precision())).device(device));
    
    tcnn::cpp::Context tmlp_ctx;
    if (!input.requires_grad() && !params.requires_grad()){
        tmlp_wp->module_->inference(stream, batch_size, input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params));
    }
    else{
        tmlp_ctx = tmlp_wp->module_->forward(stream, batch_size, input.data_ptr<float>(), 
                                            void_data_ptr(output), void_data_ptr(params),
                                            input.requires_grad());
    }
    
    // torch::cuda::synchronize()
    CHECK_EQ(output.scalar_type(), torch_type(tmlp_wp->module_->output_precision()));
    tmlp_wp->query_output_ = output;
    tmlp_wp->query_pts_ = input;
    tmlp_wp->tmlp_ctx_ = std::move(tmlp_ctx);
    return {output};
}

variable_list TMLPFunction::backward(AutogradContext *ctx,
                                    variable_list grad_output)
{
    auto info_ptr = ctx->saved_data["tmlp_info"].toCustomClass<TMLPInfo>();
    auto tmlp_wp = info_ptr->tmlp_;
    float scale = tmlp_wp->loss_scale_;

    if (!tmlp_wp->tmlp_ctx_.ctx){
        throw std::runtime_error("Module::bwd: called with invalid context. fwd likely (mistakenly) ran in reference mode.");
    }

    Tensor dL_doutput = grad_output[0] * scale;
    Tensor& input = tmlp_wp->query_pts_;
    Tensor& output = tmlp_wp->query_output_;
    Tensor params = tmlp_wp->params_.to(torch::kFloat16);

    CHECK_TS(input);
    CHECK_TS(params);
    CHECK_TS(output);
    CHECK_TS(dL_doutput);

    CHECK_EQ(input.scalar_type(), torch::kFloat32);
    CHECK_EQ(params.scalar_type(), torch_type(tmlp_wp->module_->param_precision()));
    CHECK_EQ(output.scalar_type(), torch_type(tmlp_wp->module_->output_precision()));
    CHECK_EQ(dL_doutput.scalar_type(), torch_type(tmlp_wp->module_->output_precision()));

    CHECK_EQ(input.size(1), tmlp_wp->module_->n_input_dims());
    CHECK_EQ(output.size(1), tmlp_wp->module_->n_output_dims());
    CHECK_EQ(params.size(0), tmlp_wp->module_->n_params());
    CHECK_EQ(output.size(0), input.size(0));
    CHECK_EQ(dL_doutput.size(0), input.size(0));

    // Check device
    at::Device device = input.device();
    CHECK_EQ(device, params.device());
    CHECK_EQ(device, output.device());
    CHECK_EQ(device, dL_doutput.device());

    const at::cuda::CUDAGuard device_guard(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    uint32_t batch_size = input.size(0);

    Tensor dL_dinput;
    CHECK(input.requires_grad());
    if (input.requires_grad()){
        dL_dinput = torch::empty({ batch_size, input.size(1) },
                        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    }
    torch::Tensor dL_dparams; 
    dL_dparams = torch::empty( { int(tmlp_wp->module_->n_params())}, 
                        torch::TensorOptions().dtype(torch_type(tmlp_wp->module_->param_precision())).device(device));
    tmlp_wp->module_->backward(
        stream,
        tmlp_wp->tmlp_ctx_,
        batch_size,
        input.requires_grad() ? dL_dinput.data_ptr<float>() : nullptr,
        void_data_ptr(dL_doutput),
        void_data_ptr(dL_dparams),
        input.data_ptr<float>(),
        void_data_ptr(output),
        void_data_ptr(params)
    );

    // torch::cuda::synchronize()

    // return { dL}
    dL_dinput = dL_dinput / scale;
    dL_dparams = (dL_dparams).to(torch::kFloat32) / scale;
    
    if (!torch::all(torch::isfinite(dL_dinput)).item<bool>() || 
        !torch::all(torch::isfinite(dL_doutput)).item<bool>()){
        tmlp_wp->global_data_->backward_nan_ = true;
        tmlp_wp->loss_scale_ = std::max(tmlp_wp->loss_scale_ / 2.f, 1.f);
    }

    return {dL_dinput, dL_dparams, Tensor()};
}

void TMLP::InitParams(){
    size_t seed = 19970826;
    module_->initialize_params(seed, params_.data_ptr<float>());
}

int TMLP::LoadStates(const std::vector<Tensor>& states, int idx){
    CHECK(false) << "This should be handled by the parent module";
    return idx;
}

std::vector<Tensor> TMLP::States()
{
    CHECK(false) << "This should be handled by the parent module";
    return {};
}

std::vector<torch::optim::OptimizerParamGroup> TMLP::OptimParamGroups(){
    CHECK(false) << "This should be handled by the parent module";
    return {};
}

void TMLP::Reset() {
  CHECK(false) << "This should be handled by the parent module";
}

} // namespace AutoStudio