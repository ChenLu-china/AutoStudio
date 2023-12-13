/**
* This file is part of autostudio
* Copyright (C) 
**/

#include <torch/torch.h>
#include "include/HashMap.h"


namespace AutoStudio
{
using Tensor = torch::Tensor;

TORCH_LIBRARY(dec_hash3d_vertex, m)
{
    std::cout << "register Hash3DVertexInfo" << std::endl;
    m.class_<Hash3DVertexInfo>("Hash3DVertexInfo").def(torch::init());
}

AutoStudio::Hash3DVertex::Hash3DVertex(GlobalData* global_data)
{   
    /**
     *  Using different hash numbers pi_(i, k) and offsets delta_(i, k) to achieve multiple hash function
     *  Detial in f2-nerf 3.4
     * 
    */
    std::cout << "Hash3DVertex::Hash3DVertex" << std::endl;
    global_data_ = global_data;   
    
    const auto& config = global_data->config_["models"]["fields"];
    pool_size_ = ( 1 << config["log2_table_size"].as<int>()) * N_LEVELS;  // maximum hash table size in instant-npg

    mlp_hidden_dim_ = config["mlp_hidden_dim"].as<int>();
    mlp_out_dim_ = config["mlp_out_dim"].as<int>();
    n_hidden_layers_ = config["n_hidden_layers"].as<int>();

    // feat pool
    feat_pool_ = (torch::rand({pool_size_, N_CHANNELS}, CUDAFloat) * .2f - 1.f) * 1e-4f; //hash feature table size
    feat_pool_.requires_grad_(true);
    CHECK(feat_pool_.is_contiguous());

    n_volumes_ = global_data_->n_volumes_;
    // std::cout << pool_size_ << std::endl;
    // std::cout << n_volumes_ << std::endl;

    // Get different prime numbers
    auto is_prim = [](int x){
        for (int i = 2; i * i <= x; ++i){
            if (x % i == 0) return false;
        }
        return true;
    };

    // random select prim number between min_local_prim and max_local_prim 
    std::vector<int> prim_selected;
    int min_local_prim = 1 << 28;
    int max_local_prim = 1 << 30;
    // std::cout << n_volumes_ << std::endl;
    // std::cout << 3 * N_LEVELS * n_volumes_ << std::endl;
    for (int i = 0; i < 3 * N_LEVELS * n_volumes_; ++i){
        int val;
        do {
            val = torch::randint(min_local_prim, max_local_prim, {1}, CPUInt).item<int>();
        }
        while(!is_prim(val));
        prim_selected.push_back(val);
    }
    // std::cout << prim_selected.size() << std::endl;
    CHECK_EQ(prim_selected.size(), 3 * N_LEVELS * n_volumes_);

    prim_pool_ = torch::from_blob(prim_selected.data(), 3 * N_LEVELS * n_volumes_, CPUInt).to(torch::kCUDA);
    prim_pool_ = prim_pool_.reshape({N_LEVELS, n_volumes_, 3}).contiguous();

    if (config["rand_bias"].as<bool>()) {
        bias_pool_ = (torch::rand({ N_LEVELS * n_volumes_, 3}, CUDAFloat) * 1000.f + 100.f).contiguous();
    }
    else {
        bias_pool_ = torch::zeros({N_LEVELS * n_volumes_, 3}, CUDAFloat).contiguous();
    }

    // size of each level & each volume
    {
        int local_size = pool_size_ / N_LEVELS;
        // std::cout << local_size << std::endl;
        local_size = (local_size >> 4) << 4;
        feat_local_size_ = torch::full( { N_LEVELS }, local_size, CUDAInt).contiguous();
        feat_local_idx_ = torch::cumsum(feat_local_size_, 0) - local_size;
        feat_local_idx_ = feat_local_idx_.to(torch::kInt32).contiguous();
    }

    // initial MLP
    mlp_ = std::make_unique<TMLP>(global_data, N_LEVELS * N_CHANNELS, mlp_out_dim_, mlp_hidden_dim_, n_hidden_layers_);
}


Tensor Hash3DVertex::AnchoredQuery(const Tensor& points, const Tensor& anchors)
{
#ifdef PROFILE
#endif
    auto info = torch::make_intrusive<Hash3DVertexInfo>();

    query_points_ = ((points + 1.f) * .5f).contiguous();  // [-1, 1] -> [0, 1]
    query_volume_idx_ = anchors.contiguous();
    info->hash3dvertex_ = this;
    Tensor feat = Hash3DVertexFunction::apply(feat_pool_, torch::IValue(info))[0];  // [n_points, n_levels * n_channels];
    Tensor output = mlp_->Query(feat);
    output = output;
    return output;
}


std::vector<Tensor> Hash3DVertex::States()
{
    std::vector<Tensor> ret;
    ret.push_back(feat_pool_.data());
    ret.push_back(prim_pool_.data());
    ret.push_back(bias_pool_.data());
    ret.push_back(torch::full({1}, n_volumes_, CPUInt));

    ret.push_back(mlp_->params_.data());   
    return ret;
}

int Hash3DVertex::LoadStates(const std::vector<Tensor>& states, int idx)
{   
    feat_pool_.data().copy_(states[idx++]);
    prim_pool_ = states[idx++].clone().to(torch::kCUDA).contiguous();
    bias_pool_.data().copy_(states[idx++]);
    n_volumes_ = states[idx++].item<int>();

    mlp_->params_.data().copy_(states[idx++]);
    
    return idx;
}

std::vector<torch::optim::OptimizerParamGroup> Hash3DVertex::OptimParamGroups()
{
    std::vector<torch::optim::OptimizerParamGroup> ret;
    float lr = global_data_->learning_rate_;
    {
        auto opt = std::make_unique<torch::optim::AdamOptions>(lr);
        opt->betas() = {0.9, 0.99};
        opt->eps() = 1e-15;

        std::vector<Tensor> params = { feat_pool_ };
        ret.emplace_back(std::move(params), std::move(opt));
    }
    
    {
        auto opt = std::make_unique<torch::optim::AdamOptions>(lr);
        opt->betas() = {0.9, 0.99};
        opt->eps() = 1e-15;
        opt->weight_decay() = 1e-6;
        
        std::vector<Tensor> params;
        params.push_back(mlp_->params_);
        ret.emplace_back(std::move(params), std::move(opt));
    }
    return ret;
}

void Hash3DVertex::Reset()
{
    feat_pool_.data().uniform_(-1e-2f, 1e-2f);
    mlp_->InitParams();
}

} // namespace AutoStudio