/**
* This file is part of autostudio
* Copyright (C) 
**/
#include <torch/torch.h>
#include "OctreeMap.h"

namespace AutoStudio
{
using Tensor = torch::Tensor;
namespace F =torch::nn::functional;

TORCH_LIBRARY(octree_map, m)
{
  std::cout << "register volume render info" << std::endl;
  m.class_<OctreeMapInfo>("OctreeMapInfo").def(torch::init());
}

OctreeMap::OctreeMap(GlobalData* global_data)
{
    global_data_ = global_data;
    auto config = global_data_->config_["models"];
    auto dataset = RE_INTER(Dataset*, global_data_->dataset_);

    float split_dist_thres = config["split_dist_thres"].as<float>();
    sub_div_milestones_ = config["sub_div_milestones"].as<std::vector<int>>();
    compact_freq_ = config["compact_freq"].as<int>();
    max_oct_intersect_per_ray_ = config["max_oct_intersect_per_ray"].as<int>();
    std::reverse(sub_div_milestones_.begin(), sub_div_milestones_.end());

    global_near_ = config["near"].as<float>();
    scale_by_dis_ = config["scale_by_dis"].as<bool>();
    int bbox_levels = config["bbox_levels"].as<int>();
    float bbox_side_len = (1 << (bbox_levels - 1));

    sample_l_ = config["sample_l"].as<float>();
    int max_level = config["max_level"].as<int>();

    //construct octree and make warp transformation matrix
    octree_ = std::make_unique<Octree>(max_level, bbox_side_len, split_dist_thres, dataset); 

    // visualize octree map
    VisOctree();
    global_data_ -> n_volumes_ = octree_ -> octree_trans_.size();
    // std::cout << "The number of valid leaf node is :" << octree_ -> octree_trans_.size() << std::endl;
    // std::cout << "The number of valid leaf node is :" << global_data_->n_volumes_ << std::endl;
    
    // construct hash feature
    hashmap_ = std::make_unique<Hash3DVertex>(global_data_);
    RegisterSubPipe(hashmap_.get());

    // construct sh featrure
    shader_ = std::make_unique<SHShader>(global_data_);
    RegisterSubPipe(shader_.get());

    int n_images = dataset->n_images_;
    use_app_emb_ = config["use_app_emb"].as<bool>();
    app_emb_ = torch::randn({n_images, 16}, CUDAFloat);
    app_emb_.requires_grad_(true);

    auto bg_color = config["bg_color"].as<std::string>();
    if (bg_color == "white")
        bg_color_type_ = BGColorType::white;
    else if (bg_color == "black")
        bg_color_type_ = BGColorType::black;
    else
        bg_color_type_ = BGColorType::rand_noise;
}


RenderResult OctreeMap::Render(const Tensor& rays_o, 
                            const Tensor& rays_d, 
                            const Tensor& ranges, 
                            const Tensor& emb_idx)
{
#ifdef PROFILE
#endif
    int n_rays = rays_o.sizes()[0];
    std:: cout << n_rays << std::endl;
    sample_result_ = GetSamples(rays_o, rays_d, ranges);
    int n_all_pts = sample_result_.pts.sizes()[0];
    float sampled_pts_per_ray = float(n_all_pts) / float(n_rays);
    if (global_data_->mode_ == RunningMode::TRAIN){
        global_data_->sampled_pts_per_ray_ = 
            global_data_->sampled_pts_per_ray_ * 0.9f + sampled_pts_per_ray * 0.1f;
    }
    CHECK(sample_result_.pts_idx_bounds.max().item<int>() <= n_all_pts);
    CHECK(sample_result_.pts_idx_bounds.min().item<int>() >= 0);

    Tensor bg_color;
    if (bg_color_type_ == BGColorType::white) {
        bg_color = torch::ones({n_rays, 3}, CUDAFloat);
    }
    else if (bg_color_type_ == BGColorType::rand_noise) {
        if (global_data_->mode_ == RunningMode::TRAIN) {
            bg_color = torch::rand({n_rays, 3}, CUDAFloat);
        }
        else {
            bg_color = torch::ones({n_rays, 3}, CUDAFloat) * .5f;
        }
    }
    else {
        bg_color = torch::zeros({n_rays, 3}, CUDAFloat);
    }

    if (n_all_pts <= 0){
        Tensor colors = bg_color;
        if (global_data_->mode_ == RunningMode::TRAIN) {
            global_data_->meaningful_sampled_pts_per_ray_ = global_data_->meaningful_sampled_pts_per_ray_ * 0.9f;
        }
        return {
            colors,
            torch::zeros({ n_rays, 1 }, CUDAFloat),
            torch::zeros({ n_rays }, CUDAFloat),
            Tensor(),
            torch::full({ n_rays }, 512.f, CUDAFloat),
            Tensor(),
            Tensor()
        };
    }
    CHECK_EQ(rays_o.sizes()[0], sample_result_.pts_idx_bounds.sizes()[0]);

    auto DensityAct = [](Tensor x) -> Tensor {
        const float shift = 3.f;
        return torch::autograd::TruncExp::apply(x - shift)[0];
    };

    // First, inference without gradients - early stop

    SampleResultFlex sample_result_early_stop;
    {
        torch::NoGradGuard no_grad_guard;

        Tensor pts  = sample_result_.pts;
        Tensor dirs = sample_result_.dirs;
        Tensor anchors = sample_result_.anchors.index({"...", 0}).contiguous();

        Tensor scene_feat = hashmap_->AnchoredQuery(pts, anchors);
        Tensor sampled_density = DensityAct(scene_feat.index({ Slc(), Slc(0, 1) }));

        Tensor sampled_dt = sample_result_.dt;
        Tensor sampled_t = (sample_result_.t + 1e-2f).contiguous();
        Tensor sec_density = sampled_density.index({Slc(), 0}) * sampled_dt;
        Tensor alphas = 1.f - torch::exp(-sec_density);
        Tensor idx_start_end = sample_result_.pts_idx_bounds;
        Tensor acc_density = FlexOps::AccumulateSum(sec_density, idx_start_end, false);
        Tensor trans = torch::exp(-acc_density);
        Tensor weights = trans * alphas;
        Tensor mask = trans > 1e-4f;
        Tensor mask_idx = torch::where(mask)[0];

        sample_result_early_stop.pts = sample_result_.pts.index({mask_idx}).contiguous();
        sample_result_early_stop.dirs = sample_result_.dirs.index({mask_idx}).contiguous();
        sample_result_early_stop.dt = sample_result_.dt.index({mask_idx}).contiguous();
        sample_result_early_stop.t = sample_result_.t.index({mask_idx}).contiguous();
        sample_result_early_stop.anchors = sample_result_.anchors.index({mask_idx}).contiguous();

        sample_result_early_stop.first_oct_dis = sample_result_.first_oct_dis.clone();
        sample_result_early_stop.pts_idx_bounds = FilterIdxBounds(sample_result_.pts_idx_bounds, mask);

        CHECK_EQ(sample_result_early_stop.pts_idx_bounds.max().item<int>(), sample_result_early_stop.pts.size(0));


        if (global_data_->mode_ == RunningMode::TRAIN) {
            UpdateOctNodes(sample_result_,
                                        weights.detach(),
                                        alphas.detach());

            float meaningful_per_ray = mask.to(torch::kFloat32).sum().item<float>();
            meaningful_per_ray /= n_rays;
            global_data_->meaningful_sampled_pts_per_ray_ =
                global_data_->meaningful_sampled_pts_per_ray_ * 0.9f + meaningful_per_ray * 0.1f;
        }
    }


    Tensor scene_feat, edge_feat;
    Tensor pts  = sample_result_early_stop.pts;
    Tensor dirs = sample_result_early_stop.dirs;
    Tensor anchors = sample_result_early_stop.anchors.index({"...", 0}).contiguous();
    n_all_pts = pts.size(0);

    // Feature variation loss.
    if (global_data_->mode_ == RunningMode::TRAIN) {
        const int n_edge_pts = 8192;
        auto [ edge_pts, edge_anchors ] = GetEdgeSamples(n_edge_pts);
        edge_pts = edge_pts.reshape({ n_edge_pts * 2, 3 }).contiguous();
        edge_anchors = edge_anchors.reshape({ n_edge_pts * 2 }).contiguous();

        Tensor query_pts = torch::cat({ pts, edge_pts }, 0);
        Tensor query_anchors = torch::cat({ anchors, edge_anchors }, 0);
        Tensor all_feat = hashmap_->AnchoredQuery(query_pts, query_anchors);
        scene_feat = all_feat.slice(0, 0, n_all_pts);
        edge_feat = all_feat.slice(0, n_all_pts, n_all_pts + n_edge_pts * 2).reshape({ n_edge_pts, 2, -1 });
    }
    else {
        // Query density &gra color
        scene_feat = hashmap_->AnchoredQuery(pts, anchors);  // [n_pts, feat_dim];
    }


    Tensor idx_start_end = sample_result_early_stop.pts_idx_bounds;

    Tensor sampled_density = DensityAct(scene_feat.index({ Slc(), Slc(0, 1) }));

    Tensor shading_feat = torch::cat({torch::ones_like(scene_feat.index({Slc(), Slc(0, 1)}), CUDAFloat),
                                        scene_feat.index({Slc(), Slc(1, None)})}, 1);

    if (global_data_->mode_ == RunningMode::TRAIN && use_app_emb_) {
        Tensor all_emb_idx = CustomOps::ScatterIdx(n_all_pts, sample_result_early_stop.pts_idx_bounds, emb_idx);
        shading_feat = CustomOps::ScatterAdd(app_emb_, all_emb_idx, shading_feat);
    }

    Tensor sampled_colors = shader_->Query(shading_feat, dirs);
    if (global_data_->gradient_scaling_progress_ < 1.) {
        sampled_density = CustomOps::GradientScaling(sampled_density, idx_start_end,
                                                    global_data_->gradient_scaling_progress_);
        sampled_colors = CustomOps::GradientScaling(sampled_colors, idx_start_end,
                                                    global_data_->gradient_scaling_progress_);
    }
    Tensor sampled_dt = sample_result_early_stop.dt;
    Tensor sampled_t = (sample_result_early_stop.t + 1e-2f).contiguous();
    Tensor sec_density = sampled_density.index({Slc(), 0}) * sampled_dt;
    Tensor alphas = 1.f - torch::exp(-sec_density);
    Tensor acc_density = FlexOps::AccumulateSum(sec_density, idx_start_end, false);
    Tensor trans = torch::exp(-acc_density);
    Tensor weights = trans * alphas;

    Tensor last_trans = torch::exp(-FlexOps::Sum(sec_density, idx_start_end));
    Tensor colors = FlexOps::Sum(weights.unsqueeze(-1) * sampled_colors, idx_start_end);
    colors = colors + last_trans.unsqueeze(-1) * bg_color;
    Tensor disparity = FlexOps::Sum(weights / sampled_t, idx_start_end);
    Tensor depth = FlexOps::Sum(weights * sampled_t, idx_start_end) / (1.f - last_trans + 1e-4f);

    CHECK_NOT_NAN(colors);


    std::cout << sample_result_.pts.sizes()[0] << std::endl;
    
    return { Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor() };
}


void OctreeMap::VisOctree()
{
    std::ofstream f(global_data_ -> base_exp_dir_ + "/octree.obj", std::ios::out);

    auto& octree_nodes = octree_->octree_nodes_;
    int n_nodes = octree_->octree_nodes_.size();
    for(const auto& node : octree_nodes){
        for(int st = 0; st < 8; ++st){
            Wec3f xyz = node.center_ + Wec3f(((st >> 2 & 1) - 0.5f), ((st >> 1 & 1) - 0.5f), ((st >> 0 & 1) - 0.5f)) * node.extend_len_;
            f << "v " << xyz[0] << " " << xyz[1] << " " << xyz[2] << std::endl;
        }
    }

    for(int i = 0; i< n_nodes; ++i){
        if (octree_nodes[i].trans_idx_ < 0) { continue; }
        for (int a = 0; a < 8; ++a){
            for(int b = a + 1; b < 8; ++b){
                int st = (a ^ b);
                if (st == 1 || st == 2 || st == 4){
                   f << "l " << i * 8 + a + 1 << " " << i * 8 + b + 1 << std::endl;
                }
            }
        }
    }

    f.close();
}


std::vector<torch::optim::OptimizerParamGroup> OctreeMap::OptimParamGroups() {

    std::vector<torch::optim::OptimizerParamGroup> ret;
    for (auto model : sub_models_) {
        auto cur_params = model->OptimParamGroups();
        for (const auto& para_group : cur_params){
            ret.emplace_back(para_group);
            std::cout << ret.size() <<std::endl;
        }
    }

    {
        auto opt = std::make_unique<torch::optim::AdamOptions>(global_data_->learning_rate_);
        opt->betas() = {0.9, 0.99};
        opt->eps() = 1e-15;
        opt->weight_decay() = 1e-6;

        std::vector<Tensor> params;
        params.push_back(app_emb_);
        ret.emplace_back(std::move(params), std::move(opt));
    }
    return ret;
}


std::vector<Tensor> OctreeMap::States()
{
    std::vector<Tensor> ret;
    ret.push_back(octree_->octree_nodes_gpu_);
    ret.push_back(octree_->octree_trans_gpu_);
    ret.push_back(octree_->tree_visit_cnt_);
    Tensor milestones_ts = torch::from_blob(sub_div_milestones_.data(), sub_div_milestones_.size(), CPUInt).to(torch::kCUDA);
    ret.push_back(milestones_ts);
    
    return ret;
}


int OctreeMap::LoadStates(const std::vector<Tensor>& states, int idx)
{
    octree_->octree_nodes_gpu_ = states[idx++].clone().to(torch::kCUDA).contiguous();
    octree_->octree_trans_gpu_ = states[idx++].clone().to(torch::kCUDA).contiguous();
    octree_->tree_visit_cnt_   = states[idx++].clone().to(torch::kCUDA).contiguous();
    Tensor milestones_ts       = states[idx++].clone().to(torch::kCUDA).contiguous();
    
    Tensor octree_node_cpu = octree_ ->octree_nodes_gpu_.to(torch::kCPU);
    octree_->octree_nodes_.resize(octree_node_cpu.sizes()[0] / sizeof(OctreeNode));
    std::memcpy(octree_->octree_nodes_.data(), octree_node_cpu.data_ptr(), octree_node_cpu.sizes()[0]);

    Tensor octree_trans_cpu = octree_->octree_trans_gpu_.to(torch::kCPU);
    octree_->octree_trans_.resize(octree_trans_cpu.sizes()[0] / sizeof(OctreeTransInfo));
    std::memcpy(octree_->octree_trans_.data(), octree_trans_cpu.data_ptr(), octree_trans_cpu.sizes()[0]);

    sub_div_milestones_.resize(milestones_ts.sizes()[0]);
    std::memcpy(sub_div_milestones_.data(), milestones_ts.data_ptr(), milestones_ts.sizes()[0] * sizeof(int));
    PRINT_VAL(sub_div_milestones_);

    octree_->tree_weight_stats_ = torch::full({ int(octree_->octree_nodes_.size()) }, INIT_NODE_STAT, CUDAInt);
    octree_->tree_alpha_stats_ = torch::full({ int(octree_->octree_nodes_.size()) }, INIT_NODE_STAT, CUDAInt);

    int valid_nodes = 0;
    for(int i = 0; i < octree_->octree_nodes_.size(); ++i){
        if (octree_->octree_nodes_[i].trans_idx_ >= 0){
            valid_nodes++;
        }
    }
    PRINT_VAL(valid_nodes);

    return idx;
}


} // namespace AutoStudio