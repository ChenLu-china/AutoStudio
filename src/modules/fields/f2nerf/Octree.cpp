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

Octree::Octree(int max_depth,
               float bbox_side_len,
               float split_dist_thres,
               Dataset* data_set)
{
    fmt::print("[Octree::Octree]: begin \n");
    max_depth_ = max_depth;
    bbox_len_ = bbox_side_len;
    dist_thres_ = split_dist_thres;
    train_set_ = data_set->sampler_->train_set_;
    images_ = data_set->sampler_->images_;
    c2w_ = data_set->GetTrainC2W_Tensor(true);
    w2c_ = data_set->GetTrainW2C_Tensor(true);
    intri_ = data_set->GetTrainIntri_Tensor(true);

    OctreeNode root;
    root.parent_ = -1;
    octree_nodes_.push_back(root);

    AddTreeNode(0, 0, Wec3f::Zero(), bbox_side_len);
}

inline void Octree::AddTreeNode(int u, int depth, Wec3f center, float bbox_len)
{
    /**
     * Implementation of octree construction renference section 3.3     
    */
    CHECK_LT(u, octree_nodes_.size());

    OctreeNode node;
    octree_nodes_[u].center_ = center;
    octree_nodes_[u].is_leaf_node_ = false;
    octree_nodes_[u].extend_len_ = bbox_len;
    octree_nodes_[u].trans_idx_ = -1;

    for(int i = 0; i < 8; ++i) octree_nodes_[u].child_[i] = -1;

    if (depth > max_depth_) {
        octree_nodes_[u].is_leaf_node_ = true;
        octree_nodes_[u].trans_idx_ = -1;
        return;
    }

    // calculate distance from camera to hash cube ceneter
    int num_imgs = c2w_.sizes()[0];
    std::cout << num_imgs << std::endl;
    Tensor center_hash = torch::zeros({3}, CPUFloat);
    std::memcpy(center_hash.data_ptr(), &center, 3 * sizeof(float));
    center_hash = center_hash.to(torch::kCUDA);
    
    const int n_rand_pts = 32 * 32 * 32;
    Tensor rand_pts = (torch::rand({n_rand_pts, 3}, CUDAFloat) - .5f) * bbox_len + center_hash.unsqueeze(0);
    auto visi_cams = GetVaildCams(bbox_len, center_hash);

    Tensor cam_pose_ts = c2w_.index({Slc(), Slc(0, 3), 3}).to(torch::kCUDA).contiguous();
    Tensor cam_dis = torch::linalg_norm(cam_pose_ts - center_hash.unsqueeze(0), 2, -1, true);
    cam_dis = cam_dis.to(torch::kCPU).contiguous();

    std::vector<float> visi_dis;
    for (int visi_cam : visi_cams) {
        float cur_dis = cam_dis[visi_cam].item<float>();
        visi_dis.push_back(cur_dis);
    }
    Tensor visi_dis_ts = torch::from_blob(visi_dis.data(), { int(visi_dis.size() )}, CPUFloat).to(torch::kCUDA);
    float distance_summary = DistanceSummary(visi_dis_ts);

    bool exist_unaddressed_cams = ((visi_cams.size() >= N_PROS / 2) && (distance_summary < bbox_len * dist_thres_));

    if(exist_unaddressed_cams) {
        for(int st = 0; st < 8; ++st) {
            int v = octree_nodes_.size();
            octree_nodes_.emplace_back();  //  create tree node at the end of vector<OctreeNode> different from push_back
            Wec3f offset(float((st >> 2) & 1) - .5f, float((st >> 1) & 1) - .5f, float(st & 1) - .5f);
            Wec3f sub_center = center + bbox_len * .5f * offset;
            octree_nodes_[u].child_[st] = v;
            octree_nodes_[v].parent_ = u;

            AddTreeNode(v, depth + 1, sub_center, bbox_len * 0.5f);
        }
    } else if (visi_cams.size() < N_PROS / 2) {
        octree_nodes_[u].is_leaf_node_ = true;
        octree_nodes_[u].trans_idx_ = -1;
    } else {
        octree_nodes_[u].is_leaf_node_ = true;
        octree_nodes_[u].trans_idx_ = octree_trans_.size();
        Tensor visi_cam_c2w = torch::zeros({ int(visi_cams.size()), 3, 4 }, CUDAFloat);
        Tensor visi_cam_intri = torch::zeros({int(visi_cams.size()), 3, 3}, CUDAFloat);
        for (int i = 0; i < visi_cams.size(); ++i) {
            visi_cam_c2w.index_put_({i}, c2w_.index({visi_cams[i]}));
            visi_cam_intri.index_put_({i}, intri_.index({visi_cams[i]}));
        }
        octree_trans_.push_back(addTreeTrans(rand_pts, visi_cam_c2w, visi_cam_intri, center_hash)); // calculate warp matrix reference supplementray A.2 
    }
}

float Octree::DistanceSummary(const Tensor& dis)
{
    if (dis.reshape(-1).size(0) <= 0) { return 1e8f; }
    Tensor log_dis = torch::log(dis);
    float thres = torch::quantile(log_dis, 0.25).item<float>();
    Tensor mask = (log_dis < thres).to(torch::kFloat32);
    if (mask.sum().item<float>() < 1e-3f) {
        return std::exp(log_dis.mean().item<float>());
    }
    return std::exp(((log_dis * mask).sum() / mask.sum()).item<float>());
}

std::vector<int> Octree::GetVaildCams(float bbox_len, const Tensor& center)
{   
    std::vector<Tensor> rays_o, rays_d, bounds;
    const int n_image = train_set_.sizes()[0];

    for(int i = 0; i < n_image; ++i) {   
        auto img = images_[i];
        img.toCUDA();
        float half_w = img.intri_.index({0, 2}).item<float>();
        float half_h = img.intri_.index({1, 2}).item<float>();
        int res_w = 128;
        int res_h = std::round(res_w / half_w * half_h);
        
        Tensor ii = torch::linspace(0.f, half_w * 2.f - 1.f, res_h, CUDAFloat);
        Tensor jj = torch::linspace(0.f, half_h * 2.f - 1.f, res_w, CUDAFloat);
        auto ij = torch::meshgrid({ii, jj}, "ij");

        ii = ij[0].reshape({-1});
        jj = ij[1].reshape({-1});
        Tensor ij_ = torch::stack({ii, jj}, -1).to(torch::kCUDA).contiguous();
        auto [ray_o, ray_d] = img.Img2WorldRayFlex(ij_.to(torch::kInt32));
        Tensor bound = torch::stack({
                torch::full({ 1 }, img.near_, CUDAFloat),
                torch::full({ 1 }, img.far_,  CUDAFloat)}, 
                -1).contiguous();
        bound = bound.reshape({-1, 2});
        // Tensor bound = torch::from_blob({img.near_, img.far_}, {2}, OptionFloat32);
        img.toHost();
        rays_o.push_back(ray_o);
        rays_d.push_back(ray_d);
        bounds.push_back(bound);
    }
    
    Tensor rays_o_tensor = torch::stack(rays_o, 0).reshape({n_image, -1, 3}).to(torch::kFloat32).contiguous();
    Tensor rays_d_tensor = torch::stack(rays_d, 0).reshape({n_image, -1, 3}).to(torch::kFloat32).contiguous();
    Tensor bounds_tensor = torch::stack(bounds, 0).reshape({n_image, 2}).to(torch::kFloat32).contiguous();
    std::cout << rays_o_tensor.sizes() << std::endl;
    Tensor a = ((center - bbox_len * .5f).index({None, None}) - rays_o_tensor) / rays_d_tensor;
    Tensor b = ((center + bbox_len * .5f).index({None, None}) - rays_o_tensor) / rays_d_tensor;
    a = torch::nan_to_num(a, 0.f, 1e6f, -1e6f);
    b = torch::nan_to_num(b, 0.f, 1e6f, -1e6f);
    Tensor aa = torch::maximum(a, b);
    Tensor bb = torch::minimum(a, b);
    auto [ far, far_idx ] = torch::min(aa, -1);
    auto [ near, near_idx ] = torch::max(bb, -1);
    far = torch::minimum(far, bounds_tensor.index({Slc(), None, 1}));
    near = torch::maximum(near, bounds_tensor.index({Slc(), None, 0}));
    Tensor mask = (far > near).to(torch::kFloat32).sum(-1);
    Tensor good = torch::where(mask > 0)[0].to(torch::kInt32).to(torch::kCPU);
    std::vector<int> ret;
    for (int idx = 0; idx < good.sizes()[0]; idx++) {
        ret.push_back(good[idx].item<int>());
    }
    return ret;
}

std::tuple<Tensor, Tensor> PCA(const Tensor& pts) {
  Tensor mean = pts.mean(0, true); // [n_pts, ]
  Tensor moved = pts - mean;
  Tensor cov = torch::matmul(moved.unsqueeze(-1), moved.unsqueeze(1));  // [ n_pts, n_frames, n_frames ];
  cov = cov.mean(0);
  auto [ L, V ] = torch::linalg_eigh(cov);
  L = L.to(torch::kFloat32);
  V = V.to(torch::kFloat32);
  auto [ L_sorted, indices ] = torch::sort(L, 0, true);
  V = V.permute({1, 0}).contiguous().index({ indices }).permute({1, 0}).contiguous();   // { in_dims, 3 }
  L = L.index({ indices }).contiguous();
  return { L, V };
}

OctreeTransInfo Octree::addTreeTrans(const Tensor& rand_pts,const Tensor& c2w, const Tensor& intri, const Tensor& center)
{   
    int n_virt_cams = N_PROS / 2;
    int n_cur_cams = c2w.size(0);
    int n_pts = rand_pts.size(0);

    Tensor cam_pos = c2w.index({Slc(), Slc(0, 3), 3}).contiguous();
    Tensor cam_axes = torch::linalg_inv(c2w.index({Slc(), Slc(0, 3), Slc(0, 3)})).contiguous();

    // First step: align distance, find good cameras
    Tensor dis = torch::linalg_norm(cam_pos - center.unsqueeze(0), 2, -1, false);
    float dis_summary = DistanceSummary(dis);

    Tensor rel_cam_pos, normed_cam_pos;

    rel_cam_pos = (cam_pos - center.unsqueeze(0)) / dis.unsqueeze(-1) * dis_summary;
    normed_cam_pos = (cam_pos - center.unsqueeze(0)) / dis.unsqueeze(-1);

    Tensor dis_pairs = torch::linalg_norm(normed_cam_pos.unsqueeze(0) - normed_cam_pos.unsqueeze(1), 2, -1, false);
    dis_pairs = dis_pairs.to(torch::kCPU).contiguous();
    const float* dis_pairs_ptr = dis_pairs.data_ptr<float>();

    std::vector<int> good_cams;
    std::vector<int> cam_marks(n_cur_cams);
    CHECK_GT(n_cur_cams, 0);
    good_cams.push_back(torch::randint(n_cur_cams, {1}, CPUInt).item<int>());
    cam_marks[good_cams[0]] = 1;

    for (int cnt = 1; cnt < n_virt_cams && cnt < n_cur_cams; cnt++) {
        int candi = -1; float max_dis = -1.f;
        for (int i = 0; i < n_cur_cams; i++) {
            if (cam_marks[i]) continue;
            float cur_dis = 1e8f;
            for (int j = 0; j < n_cur_cams; j++) {
                if (cam_marks[j]) cur_dis = std::min(cur_dis, dis_pairs_ptr[i * n_cur_cams + j]);
            }
            if (cur_dis > max_dis) {
                max_dis = cur_dis;
                candi = i;
            }
        }
        CHECK_GE(candi, 0);
        cam_marks[candi] = 1;
        good_cams.push_back(candi);
    }

    // In case where there are not enough cameras
    for (int i = 0; good_cams.size() < n_virt_cams; i++) {
        good_cams.push_back(good_cams[i]);
    }

    // Second step: Construct pers trans
    // At GPU
    Tensor good_cam_scale = torch::ones({ n_virt_cams }, CUDAFloat);
    Tensor good_cam_pos = torch::zeros({ n_virt_cams, 3 }, CUDAFloat);
    Tensor good_rel_cam_pos = torch::zeros({ n_virt_cams, 3 }, CUDAFloat);
    Tensor good_cam_axis = torch::zeros({ n_virt_cams, 3, 3 }, CUDAFloat);

    CHECK_EQ(good_cams.size(), n_virt_cams);

    Tensor cam_scale = (dis / dis_summary).clip(1.f, 1e9f);
    rel_cam_pos = (cam_pos - center.unsqueeze(0)) / dis.unsqueeze(-1) * dis.unsqueeze(-1).clip(dis_summary, 1e9f);
    for (int i = 0; i < good_cams.size(); i++) {
        good_cam_pos.index_put_({i}, (rel_cam_pos[good_cams[i]] + center));
        good_rel_cam_pos.index_put_({i}, (rel_cam_pos[good_cams[i]]));
        good_cam_axis.index_put_({i}, cam_axes[good_cams[i]]);
        good_cam_scale.index_put_({i}, cam_scale[good_cams[i]]);
    }

    Tensor expect_z_axis = good_rel_cam_pos / torch::linalg_norm(good_rel_cam_pos, 2, -1, true);
    Tensor rots = torch::zeros({ n_virt_cams, 3, 3 }, CUDAFloat);

    auto ToEigenVec3 = [](Tensor x) {
        Wec3f ret;
        x = x.to(torch::kCPU);
        for (int i = 0; i < 3; i++) {
            ret(i) = x[i].item<float>();
        }
        return ret;
    };

    auto ToTorchMat33 = [](Watrix33f x) {
        Tensor ret = torch::zeros({3, 3}, CPUFloat);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                ret.index_put_({i, j}, x(i, j));
            }
        }
        return ret.to(torch::kCUDA);
    };

    for (int i = 0; i < good_cams.size(); i++) {
        Wec3f from_z_axis = ToEigenVec3(good_cam_axis.index({i, 2, Slc(0, 3)}));
        Wec3f to_z_axis = ToEigenVec3(expect_z_axis.index({i, Slc(0, 3)}));
        Wec3f crossed = from_z_axis.cross(to_z_axis);
        float cos_val = from_z_axis.dot(to_z_axis);
        float sin_val = crossed.norm();
        float angle = std::asin(sin_val);
        if (cos_val < 0.f) {
            angle = M_PI - angle;
        }
        crossed = crossed.normalized();
        Watrix33f rot_mat;
        rot_mat = Eigen::AngleAxisf(angle, crossed);

        rots.index_put_({ i }, ToTorchMat33(rot_mat));
    }

    good_cam_axis = torch::matmul(good_cam_axis, rots.transpose(1, 2));

    Tensor x_axis = good_cam_axis.index({Slc(), 0, Slc()}).contiguous();
    Tensor y_axis = good_cam_axis.index({Slc(), 1, Slc()}).contiguous();
    Tensor z_axis = good_cam_axis.index({Slc(), 2, Slc()}).contiguous();
    Tensor diff = z_axis - expect_z_axis;
    CHECK_LT(diff.abs().max().item<float>(), 1e-3f);

    float focal = (intri.index({ Slc(), 0, 0 }) / intri.index({ Slc(), 0, 2 })).item<float>();
    x_axis *= focal; y_axis *= focal;
    x_axis *= good_cam_scale.unsqueeze(-1); y_axis *= good_cam_scale.unsqueeze(-1);
    x_axis = torch::cat({x_axis, y_axis}, 0);
    z_axis = torch::cat({z_axis, z_axis}, 0);

    Tensor wp_cam_pos = torch::cat({good_cam_pos, good_cam_pos}, 0);
    Tensor frame_trans = torch::zeros({N_PROS, 2, 4}, CUDAFloat);
    frame_trans.index_put_({Slc(), 0, Slc(0, 3)}, x_axis);
    frame_trans.index_put_({Slc(), 1, Slc(0, 3)}, z_axis);
    frame_trans.index_put_({Slc(), 0, 3}, -(x_axis * wp_cam_pos).sum(-1));
    frame_trans.index_put_({Slc(), 1, 3}, -(z_axis * wp_cam_pos).sum(-1));

    // Third step: Construct frame weight by PCA.
    // Mapped points and Jacobian
    Tensor transed_pts = torch::matmul(frame_trans.index({ None, Slc(), Slc(), Slc(0, 3)}), rand_pts.index({ Slc(), None, Slc(), None}));
    transed_pts = transed_pts.index({"...", 0}) + frame_trans.index({ None, Slc(), Slc(), 3 });

    Tensor dv_da = 1.f / transed_pts.index({Slc(), Slc(), 1 });
    Tensor dv_db = transed_pts.index({Slc(), Slc(), 0 }) / -transed_pts.index({Slc(), Slc(), 1 }).square();
    Tensor dv_dab = torch::stack({ dv_da, dv_db }, -1); // [ n_pts, N_PROS, 2 ]
    Tensor dab_dxyz = frame_trans.index({ None, Slc(), Slc(), Slc(0, 3)}).clone(); // [ n_pts, N_PROS, 2, 3 ];
    Tensor dv_dxyz = torch::matmul(dv_dab.unsqueeze(2), dab_dxyz).index({Slc(), Slc(), 0, Slc()});  // [ n_pts, N_PROS, 3 ];

    CHECK(transed_pts.index({Slc(), Slc(), 1 }).max().item<float>() < 0.f);
    transed_pts = transed_pts.index({Slc(), Slc(), 0 }) / transed_pts.index({Slc(), Slc(), 1 });

    CHECK_NOT_NAN(transed_pts);

    // Cosntruct lin mapping
    Tensor L, V;
    std::tie(L, V) = PCA(transed_pts);
    V = V.permute({1, 0}).index({Slc(0, 3)}).contiguous(); // [ 3, N_PROS ]

    Tensor jac = torch::matmul(V.index({None}), dv_dxyz);   // [ n_pts, 3, 3 ];
    Tensor jac_warp2world = torch::linalg_inv(jac);
    Tensor jac_warp2image = torch::matmul(dv_dxyz, jac_warp2world);

    Tensor jac_abs = jac_warp2image.abs();  // [n_pts, N_PROS, 3]
    auto [ jac_max, max_tmp ] = torch::max(jac_abs, 1); // [ n_pts, 3 ]
    Tensor exp_step = 1.f / jac_max;  // [n_pts, 3];
    Tensor mean_step = exp_step.mean(0);
    V /= mean_step.unsqueeze(-1);

    Tensor V_cpu = V.to(torch::kCPU).contiguous();
    Tensor frame_trans_cpu = frame_trans.to(torch::kCPU).contiguous();

    CHECK_NOT_NAN(V_cpu);
    CHECK_NOT_NAN(frame_trans_cpu);
    OctreeTransInfo ret;
    std::memcpy(&(ret.w2xz), frame_trans_cpu.data_ptr(), sizeof(PersMatType) * N_PROS);
    std::memcpy(ret.weight.data(), V_cpu.data_ptr(), sizeof(TransWetType));
    for (int i = 0; i < 3; i++) {
        ret.center[i] = center[i].item<float>();
    }
    ret.dis_summary = dis_summary;
    return ret;
}

} // namespace AutoStudio