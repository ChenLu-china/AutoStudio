/**
* This file is part of autostudio
* Copyright (C) 
**/
#include <torch/torch.h>
#include <fmt/core.h>
#include "Octree.h"
#include "../../../dataset/Dataset.h"
#include "../../camera_manager/Image.h"

namespace AutoStudio
{
using Tensor = torch::Tensor;

AutoStudio::Octree::Octree(int max_depth, float bbox_side_len, float split_dist_thres,
            Dataset* data_set)
{
    fmt::print("[Octree::Octree]: begin");
    max_depth_ = max_depth;
    bbox_len_ = bbox_side_len;
    dist_thres_ = split_dist_thres;
    c2w_ = 
    train_set_ = data_set->train_set_;
    images_ = data_set->sampler_->images_;


    OctreeNode root;
    root.parent_ = -1;
    octree_nodes_.push_back(root);

    AddTreeNode(0, 0, Wec3f::Zero(), bbox_side_len);

}

inline void Octree::AddTreeNode(int u, int depth, Wec3f center, float bbox_len){
    CHECK_LT(u, octree_nodes_.size());

    OctreeNode node;
    octree_nodes_[u].center_ = center;
    octree_nodes_[u].is_leaf_node_ = false;
    octree_nodes_[u].extend_len_ = bbox_len;
    octree_nodes_[u].trans_idx_ = -1;

    for(int i = 0; i < 8; ++i) octree_nodes_[u].child_[i] = -1;

    if(depth > max_depth_){
        octree_nodes_[u].is_leaf_node_ = true;
        octree_nodes_[u].trans_idx_ = -1;
        return;
    }

    // calculate distance from camera to hash cube ceneter
    int num_imgs = c2w_.sizes()[0];
    Tensor center_hash = torch::zeros({3}, CPUFloat);
    std::memcpy(center_hash.data_ptr(), &center, 3 * sizeof(float));
    center_hash = center_hash.to(torch::kCUDA);
    
    const int n_rand_pts = 32 * 32 * 32;
    Tensor rand_pts = (torch::rand({n_rand_pts, 3}, CUDAFloat) - .5f) * bbox_len + center_hash.unsqueeze(0);
    auto visi_cams = GetVaildCams(bbox_len, center_hash);



}

std::vector<int> Octree::GetVaildCams(float bbox_len, 
                               const Tensor& center)
{   

    std::vector<Tensor> rays_o, rays_d, bounds;
    const int n_image = train_set_.sizes()[0];

    for(int i = 0; i < n_image; ++i)
    {   
        auto img = images_[i];
        img.toCUDA();
        float half_w = img.intri_.index({Slc(), 0, 2}).item<float>();
        float half_h = img.intri_.index({Slc(), 1, 2}).item<float>();
        int res_w = 128;
        int res_h = std::round(res_w / half_w * half_h);
        
        Tensor ii = torch::linspace(0.f, half_w * 2.f - 1.f, res_h, CUDAFloat);
        Tensor jj = torch::linspace(0.f, half_h * 2.f - 1.f, res_w, CUDAFloat);
        auto ij = torch::meshgrid({ii, jj}, "ij");

        Tensor ii = ij[0].reshape({-1});
        Tensor jj = ij[1].reshape({-1});
        Tensor ij_ = torch::stack({ii, jj}, -1).to(torch::kCUDA).contiguous();
        auto [ray_o, ray_d] = img.Img2WorldRayFlex(ij_.to(torch::kInt32));
        image.toHost();
        rays_o.push_back(ray_o);
        rays_d.push_back(ray_d);
        bounds.push_back(torch::from_blob({img.near_, img.far_}));
    }
    Tensor rays_o_tensor = torch::stack(rays_o, 0).reshape({n_image, -1, 3}).to(torch::kFloat32).contiguous();
    Tensor rays_d_tensor = torch::stack(rays_d, 0).reshape({n_image, -1, 3}).to(torch::kFloat32).contiguous();
    Tensor bounds_tensor = torch::stack(bounds, 0).reshape({n_image, 2}).to(torch::kFloat32).contiguous();
    Tensor a = ((center - bbox_len * .5f).index({None, None}) - rays_o_tensor) / rays_d_tensor;
    Tensor b = ((center - bbox_len * .5f).index({None, None}) - rays_o_tensor) / rays_d_tensor;
    a = torch::nan_to_num(a, 0.f, 1e6f, -1e6f);
    b = torch::nan_to_num(b, 0.f, 1e6f, -1e6f);
    Tensor aa = torch::maximum(a, b);
    Tensor bb = torch::minimum(a, b);
    auto [ far, far_idx ] = torch::min(aa, -1);
    auto [ near, near_idx ] = torch::max(bb, -1);
    far = torch::minimum(far, bound.index({Slc(), None, 1}));
    near = torch::maximum(near, bound.index({Slc(), None, 0}));
    Tensor mask = (far > near).to(torch::kFloat32).sum(-1);
    Tensor good = torch::where(mask > 0)[0].to(torch::kInt32).to(torch::kCPU);
    std::vector<int> ret;
    for (int idx = 0; idx < good.sizes()[0]; idx++) {
        ret.push_back(good[idx].item<int>());
    }
    return ret;
}

} // namespace AutoStudio