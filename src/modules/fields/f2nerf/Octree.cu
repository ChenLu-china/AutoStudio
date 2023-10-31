/**
* This file is part of autostudio
* Copyright (C) 
**/

#include "Octree.h"

namespace AutoStudio
{


__device__ int CheckVisible(const Wec3f& center, float side_len,
                            const Watrix33f& intri, const Watrix34f& w2c, const Wec2f& bound) {
  Wec3f cam_pt = w2c * center.homogeneous();
  float radius = side_len * 0.707;
  if (-cam_pt.z() < bound(0) - radius ||
      -cam_pt.z() > bound(1) + radius) {
    return 0;
  }
  if (cam_pt.norm() < radius) {
    return 1;
  }

  float cx = intri(0, 2);
  float cy = intri(1, 2);
  float fx = intri(0, 0);
  float fy = intri(1, 1);
  float bias_x = radius / -cam_pt.z() * fx;
  float bias_y = radius / -cam_pt.z() * fy;
  float img_pt_x = cam_pt.x() / -cam_pt.z() * fx;
  float img_pt_y = cam_pt.y() / -cam_pt.z() * fy;
  if (img_pt_x + bias_x < -cx || img_pt_x > cx + bias_x ||
      img_pt_y + bias_y < -cy || img_pt_y > cy + bias_y) {
    return 0;
  }
  return 1;
}


__global__ void MarkInvisibleNodesKernel(int n_nodes, int n_cams,
                                         OctreeNode* octree_nodes,
                                         Watrix33f* intris, Watrix34f* w2cs, Wec2f* bounds) {
  int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (node_idx >= n_nodes) { return; }
  int n_visible_cams = 0;
  for (int cam_idx = 0; cam_idx < n_cams; cam_idx++) {
    n_visible_cams += CheckVisible(octree_nodes[node_idx].center_,
                                   octree_nodes[node_idx].extend_len_,
                                   intris[cam_idx],
                                   w2cs[cam_idx],
                                   bounds[cam_idx]);
  }
  if (n_visible_cams < 1) {
    octree_nodes[node_idx].trans_idx_ = -1;
  }
}


void Octree::MarkInvisibleNodes() {
  int n_nodes = octree_nodes_.size();
  int n_cams = intri_.size(0);

  CK_CONT(intri_);
  CK_CONT(w2c_);
  CK_CONT(bound_);

  dim3 block_dim = LIN_BLOCK_DIM(n_nodes);
  dim3 grid_dim = LIN_GRID_DIM(n_nodes);
  MarkInvisibleNodesKernel<<<grid_dim, block_dim>>>(
      n_nodes, n_cams,
      RE_INTER(OctreeNode*, octree_nodes_gpu_.data_ptr()),
      RE_INTER(Watrix33f*, intri_.data_ptr()),
      RE_INTER(Watrix34f*, w2c_.data_ptr()),
      RE_INTER(Wec2f*, bound_.data_ptr())
  );
}

} // namespace AutoStudio