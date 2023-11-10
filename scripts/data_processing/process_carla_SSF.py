
"""
Run this python file will do follow works:
    Step 1: Normalize camera poses 
    Step 2: Load z-buffer depth or change it to depth distance
    Step 3: Calculate point clouds and frustum cameras
    Step 4: Visualize them use open3d
    Step 5: Save data used in NeRF's trainning
"""

# Copyright 2023 Lu Chen, reference 2020 NeRF++ create by Kai Zhang


import os
import cv2
import math
import json
import torch
import pickle
import argparse
import numpy as np
import open3d as o3d
import skimage.io as io
import imageio.v2 as imageio
import torch.nn.functional as F

from PIL import Image
from skimage import img_as_ubyte
from tqdm import tqdm
from numpy.matlib import repmat
from matplotlib import pyplot as plt

# from pyvirtualdisplay import Display
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
from typing import Optional
from pathlib import Path
from collections import Counter
from kornia.core import Tensor, stack
from kornia.utils._compat import torch_meshgrid
# from dataLoader.ray_utils import rays_intersect_sphere
# from dataset.preprocessing import visualize_depth_numpy
from utils import post_prediction_SSF, visualize_depth

import sys
sys.path.append(os.path.realpath('./scripts'))
print(sys.path)
from lib_tools.depth_normals.mini_omnidata import OmnidataModel

os.environ['CURL_CA_BUNDLE'] = ''
# display = Display(visible=True, size=(2560, 1440))
# display.start()
# import os 
# os.environ["QT_API"] = "pyqt"
# from mayavi import mlab


# mlab.options.offscreen = True

# figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))

cam_infos = {
    "front_120": {"fov":120, "x": 2.3, "y": 0, "z": 1.2, "yaw": 0.0},
    "leftfront_100": {"fov":100, "x": 1.1, "y": -1.1, "z": 1.2, "yaw": -45.0},
    "rightfront_100": {"fov":100, "x": 1.1, "y": 1.1, "z": 1.2, "yaw": 45.0},
    "back_100": {"fov":100, "x": -2.3, "y": 0.0, "z": 1.2, "yaw": 180.0},
    "leftback_100": {"fov":100, "x": 1.3, "y": -1.1, "z": 1.2, "yaw": -133.0}, 
    "rightback_100": {"fov":100, "x": 1.3, "y": 1.1, "z": 1.2, "yaw": 133.0}  
}

world_offset = np.array([  1.21562983, -44.79893401, -20.17243147])

def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Generate a coordinate grid for an image.

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.

    Args:
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.

    Return:
        grid tensor with shape :math:`(1, H, W, 2)`.

    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[-1., -1.],
                  [ 1., -1.]],
        <BLANKLINE>
                 [[-1.,  1.],
                  [ 1.,  1.]]]])

        >>> create_meshgrid(2, 2, normalized_coordinates=False)
        tensor([[[[0., 0.],
                  [1., 0.]],
        <BLANKLINE>
                 [[0., 1.],
                  [1., 1.]]]])
    """
    xs: Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys: Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    # Fix TracerWarning
    # Note: normalize_pixel_coordinates still gots TracerWarning since new width and height
    #       tensors will be generated.
    # Below is the code using normalize_pixel_coordinates:
    # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
    # if normalized_coordinates:
    #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
    # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    # TODO: torchscript doesn't like `torch_version_ge`
    # if torch_version_ge(1, 13, 0):
    #     x, y = torch_meshgrid([xs, ys], indexing="xy")
    #     return stack([x, y], -1).unsqueeze(0)  # 1xHxWx2
    # TODO: remove after we drop support of old versions
    base_grid: Tensor = stack(torch_meshgrid([xs, ys], indexing="ij"), dim=-1)  # WxHx2
    return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2

def get_ray_directions_use_intrinsics(height, width, intrinsics):
    i, j = create_meshgrid(height, width, normalized_coordinates=False)[0].unbind(-1)
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    directions = torch.stack([
        (i - cx) / fx, (j - cy) / fy, torch.ones_like(i)
    ], -1)
    return directions

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def get_matrix(translation, rotation, scale):
        matrix = np.matrix(np.identity(4))
        cy = math.cos(np.radians(rotation[2]))
        sy = math.sin(np.radians(rotation[2]))
        cr = math.cos(np.radians(rotation[0]))
        sr = math.sin(np.radians(rotation[0]))
        cp = math.cos(np.radians(rotation[1]))
        sp = math.sin(np.radians(rotation[1]))
        matrix[0, 3] = translation[0]
        matrix[1, 3] = translation[1]
        matrix[2, 3] = translation[2]
        matrix[0, 0] = scale[0] * (cp * cy)
        matrix[0, 1] = scale[1] * (cy * sp * sr - sy * cr)
        matrix[0, 2] = -scale[2] * (cy * sp * cr + sy * sr)
        matrix[1, 0] = scale[0] * (sy * cp)
        matrix[1, 1] = scale[1] * (sy * sp * sr + cy * cr)
        matrix[1, 2] = scale[2] * (cy * sr - sy * sp * cr)
        matrix[2, 0] = scale[0] * (sp)
        matrix[2, 1] = -scale[1] * (cp * sr)
        matrix[2, 2] = scale[2] * (cp * cr)
        return matrix


def ego_txt2list(ego_file):
    ego_data = []
    with open(ego_file, 'r') as f:
        for line in f.readlines():
            datas = line.strip("\n").split(" ")
        
            temp_data = []
            for data in datas:
                temp = data.split(":", 1)
                temp_data.append(float(temp[-1]))

            ego_data.append(temp_data)
    return ego_data


def to_unreal_matrix(
    to_unreal_location = [0.0, 0.0, 0.0],
    to_unreal_rotation = [-90.0, 0.0, 90.0],
    to_unreal_scale = [-1.0, 1.0, 1.0]):

    return get_matrix(to_unreal_location, to_unreal_rotation, to_unreal_scale)

def depth2array(depth):
    bgr_depth = depth[..., ::-1]
    array = bgr_depth.astype(np.float32)
    normalized_depth = np.dot(array[:,:,:3], [256.0 * 256.0, 256.0, 1.0])
    normalized_depth /= (256.0 * 256.0 * 256.0 - 1.0)
    return normalized_depth # * 1000.0

def depth2localpcd(depth, k, scale, color=None, max_depth=0.05):

    H, W = depth.shape
    depth = depth.reshape(-1)
    far = 1000.0 * scale
    # far = 40.0

    pixel_length = W * H
    # u = repmat(np.r_[W - 1 : -1 : -1], H, 1).reshape(pixel_length)
    # v = repmat(np.c_[H - 1 : -1 : -1], 1, W).reshape(pixel_length)

    u = repmat(np.r_[0 : W : 1], H, 1) # .reshape(pixel_length)
    v = repmat(np.c_[0 : H : 1], 1, W) # .reshape(pixel_length)
    depth = np.reshape(depth, pixel_length)
    max_depth_indexes = np.where(depth > max_depth)
    depth = np.delete(depth, max_depth_indexes)
    u = np.delete(u, max_depth_indexes)
    v = np.delete(v, max_depth_indexes)

    if color is not None:
        color = np.reshape(color, (pixel_length, 3))
        color = np.delete(color, max_depth_indexes, axis=0)

    p2d = np.array([u, v, np.ones_like(u)])
    inv_k = np.linalg.inv(k)
    p3d = np.dot(inv_k, p2d) * depth * far
    p3d = np.transpose(p3d)
    return p3d, color  

def calculate_pc_distance(rays_o, rays_d, k, normalized_depth, scale, max_depth=0.05):
    # H, W = img_size
    far = 1000.0 * scale

    z_depth = normalized_depth.reshape(-1)
    max_depth_indexes = np.where(z_depth <= max_depth)
    z_depth = z_depth[max_depth_indexes]
    rays_o = rays_o[max_depth_indexes]
    rays_d = rays_d[max_depth_indexes]

    d_dist = np.linalg.norm(rays_d, axis=-1)
    distance = z_depth * far * d_dist

    viewdirs = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    pcd = rays_o + distance[:, np.newaxis] * viewdirs
    return pcd

def calculate_pc_zbuff(pose, k, normalized_depths, scale):

    p3d, _ = depth2localpcd(normalized_depths, k, scale)

    p3d = np.transpose(p3d)

    p3d = np.append(p3d, np.ones((1, p3d.shape[1])), axis=0)

    word_p3d = np.dot(pose, p3d)

    p3d = np.transpose(word_p3d[0:3])
    
    return p3d

def get_camera_frustum(img_size, intrinsic, c2w, frustum_length=0.5, color=[0., 0., 1.]):
    H, W = img_size
    hfov = np.rad2deg(np.arctan(W / 2 / intrinsic[0, 0]) * 2.0)
    vfov = np.rad2deg(np.arctan(H / 2 / intrinsic[1, 1]) * 2.0)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.0))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.0))

    frustum_points = np.array([[0, 0, 0],
                              [-half_w, -half_h, frustum_length], # top-left image corner
                              [half_w, -half_h, frustum_length],  # top-right image corner
                              [-half_w, half_h, frustum_length],  # bottom-left image corner
                              [half_w, half_h, frustum_length]])  # bottom-right image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))
    # c2w = np.linalg.inv(w2c)
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), c2w.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]
    return frustum_points, frustum_lines, frustum_colors

def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset

def get_tf_cams(c2ws, target_radius=1.):
    cam_centers = []
    for c2w in c2ws:
        # W2C = np.array(cam_dict[im_name]['W2C']).reshape((4, 4))
        # C2W = np.linalg.inv(W2C)
        cam_centers.append(c2w[:3, 3:4])

    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.5

    translate = -center
    scale = target_radius / radius

    # using SSF world offset for translation 
    translate = -c2ws[0, :3, 3]
    global world_offset 
    if world_offset is None:
        world_offset = -translate
    return translate, scale

def normalize_cam_dict(c2ws, target_radius=1., in_geometry_file=None, out_geometry_file=None):
    # with open(in_cam_dict_file) as fp:
    #     in_cam_dict = json.load(fp)

    translate, scale = get_tf_cams(c2ws, target_radius=target_radius)

    if in_geometry_file is not None and out_geometry_file is not None:
        # check this page if you encounter issue in file io: http://www.open3d.org/docs/0.9.0/tutorial/Basic/file_io.html
        geometry = o3d.io.read_triangle_mesh(in_geometry_file)
        
        tf_translate = np.eye(4)
        tf_translate[:3, 3:4] = translate
        tf_scale = np.eye(4)
        tf_scale[:3, :3] *= scale
        tf = np.matmul(tf_scale, tf_translate)

        geometry_norm = geometry.transform(tf)
        o3d.io.write_triangle_mesh(out_geometry_file, geometry_norm)
  
    def transform_pose(c2w, translate, scale):
        # C2W = np.linalg.inv(W2C)
        cam_center = c2w[:3, 3]
        cam_center = (cam_center + translate) # * scale  not do rescale
        c2w[:3, 3] = cam_center
        return c2w # np.linalg.inv(C2W)
    

    norm_c2w = []
    for c2w in c2ws:
        norm_c2w.append(transform_pose(c2w, translate, scale))
        w2c = np.linalg.inv(norm_c2w[-1])
    
    return np.stack(norm_c2w, 0), translate, scale
    # out_cam_dict = copy.deepcopy(in_cam_dict)
    # for img_name in out_cam_dict:
    #     W2C = np.array(out_cam_dict[img_name]['W2C']).reshape((4, 4))
    #     W2C = transform_pose(W2C, translate, scale)
    #     assert(np.isclose(np.linalg.det(W2C[:3, :3]), 1.))
    #     out_cam_dict[img_name]['W2C'] = list(W2C.flatten())

    # with open(out_cam_dict_file, 'w') as fp:
    #     json.dump(out_cam_dict, fp, indent=2, sort_keys=True)

def norm_poses(root_dir, cam_name, num_img, max_depth=None, img_size=None, vis_normCamera:bool=False):
    sample_indices = list(range(num_img))
    
    cam_info = cam_infos[cam_name]
    fov = cam_info['fov']

    c2e_location = [float(cam_info['x']), float(cam_info['y']), float(cam_info['z'])]
    c2e_rotation = [0.0, 0.0, float(cam_info['yaw'])]
    c2e_scale = [1.0, 1.0, 1.0]
    cam2car = get_matrix(c2e_location, c2e_rotation, c2e_scale)
    
    to_unreal = to_unreal_matrix()
    cam2car = np.dot(cam2car, to_unreal)
    carla2opengl = np.eye(4)
    # carla2opengl[[1, 2]] = carla2opengl[[2, 1]]
    # carla2opengl[2, :] = -carla2opengl[2, :]
    # carla2opengl[1, :] = -carla2opengl[1, :]
    # carla2open3d = np.eye(4)
    # carla2open3d[[1, 2]] = carla2open3d[[2, 1]]
    # carla2open3d[2, :] = -carla2open3d[2, :]
    # lhand2rhand = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    change_mat = np.eye(4)
    change_mat[0, 0] = -1
    change_mat[1, 1] = -1
    ego_file = os.path.join(root_dir, "record.txt")
    ego_info = ego_txt2list(ego_file)
    # e2ws = [get_matrix(ego_info[i][1:4], ego_info[i][4:7], [1.0, 1.0, 1.0]) for i in range(len(ego_info))]
    
    dims, poses, intrinsics, intrinsics_rz = [], [], [], []
    img_h, img_w = np.array(Image.open(root_dir / f"color_{cam_name}" / f"0.png")).shape[:2]
    
    for index in sample_indices:

        intrinsic = np.identity(3, dtype=np.float32)
        intrinsic[0, 2] = img_w / 2.0
        intrinsic[1, 2] = img_h / 2.0
        intrinsic[0, 0] = intrinsic[1, 1] = img_w / (2.0 * math.tan(fov * math.pi / 360.0))
        intrinsic_rz = torch.from_numpy(np.diag([img_size[1] / img_w , img_size[0] / img_h, 1]) @ intrinsic).float()

        e2w = np.array(get_matrix(ego_info[index][1:4], ego_info[index][4:7], [1.0, 1.0, 1.0]), dtype=np.float32)
        c2w = np.dot(e2w, cam2car)
        c2w = carla2opengl @ c2w   @ change_mat
        c2w[1, 0:3] *= -1
        c2w[1, 3] *= -1
        
        intrinsics.append(torch.from_numpy(intrinsic).float())       
        dims.append([img_h, img_w])
        poses.append(torch.from_numpy(c2w).float())
        intrinsics_rz.append(intrinsic_rz)
    
    norm_poses, translate, scale = normalize_cam_dict(torch.stack(poses).numpy())
    # s2n = compute_world2normscene(torch.Tensor(dims).float(),
    #                               torch.stack(intrinsics).float(),
    #                               torch.stack(poses).float(),
    #                               max_depth=max_depth,
    #                               rescale_factor=1.0)
    # c2n = []
    # for index in sample_indices:
    #     c2n.append(s2n @ poses[index])
    # c2n = torch.stack(c2n).float()

    # camera_norm_dict = {"Norm_Scale": scale}
    if vis_normCamera:
        for index in sample_indices:
            frame = {f"{index}.png":{"K": intrinsics_rz[index].tolist(), "C2W":norm_poses[index].tolist(), "img_size":img_size}}
            camera_norm_dict.update(frame)
        saveFile = root_dir / "camera_dict_norm.json"
        with open(saveFile, 'w') as f:
            json.dump(camera_norm_dict, f, ensure_ascii=False, indent=4)
        f.close()

    return torch.from_numpy(norm_poses).float(), torch.stack(intrinsics_rz).float() # , translate, scale


def show_pc(p3d, frustums):
    coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=np.array([0., 0., 0.]))
    ## ad a single sphere
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1., resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p3d)
    o3d.visualization.draw_geometries([pcd] + [coord_mesh] + [frustums] + [sphere])
    return


def read_mate(root_dir, cam_name, img_size, max_depth, show_pc:bool = False, vis_rgbd:bool = False):
    
    color_dir = root_dir  / f"color_{cam_name}"
    depth_dir = root_dir  / f"depth_{cam_name}"
    pose_dir  = root_dir  / f"pose"
    
    num_img = len(os.listdir(color_dir))

    
    c2ws, intrinsics = norm_poses(root_dir, cam_name, num_img, max_depth=max_depth, img_size=img_size)
    # norm_scale = s2n[0, 0]
    
    all_points, max_show = [], 5
    frustums = []
    all_rgb, all_depth, all_depth_save = [], [], []
    for index in tqdm(torch.arange(0, num_img, 1), desc=f"Loading data ({num_img})"):

        img = Image.open(color_dir / f"{index}.png")
        img = torch.from_numpy(np.array(img.resize(img_size[::-1], Image.LANCZOS)) / 255.0).float()
        all_rgb += [img]

        c2w = c2ws[index]
        intrinsic = intrinsics[index]

        directions = get_ray_directions_use_intrinsics(img_size[0], img_size[1], intrinsic.numpy())
        rays_o, rays_d = get_rays(directions, c2w)

        # z_depth = Image.open(depth_dir / f'{index}.png')  # z-buffer depth
        # z_depth = depth2array(np.array(z_depth, dtype=np.float32))
        
        if show_pc:
            if index <= max_show and show_pc:
                pcd = calculate_pc_distance(rays_o.numpy(), rays_d.numpy(), intrinsic, z_depth, scale=scale)
                # pcd = calculate_pc_zbuff(c2w, intrinsic, z_depth, scale)
                frustums.append(get_camera_frustum(img_size, intrinsic, c2w, frustum_length=scale))
                all_points.append(pcd)
            else:
                cameras = frustums2lineset(frustums)
                all_points = np.concatenate(all_points, 0).reshape(-1, 3)
                # all_points = all_points
                show_pc(all_points, cameras)
        
        # caculate sphere intersections
        # sphere_intersection_displacement = rays_intersect_sphere(rays_o, rays_d, r=1)  # fg is in unit sphere

        # do pre-processing we need to process depth
        # ori_depth = z_depth.copy().reshape(-1)
        # rescaled_depth = ori_depth * 1000.0 * scale
        # dist_d = np.linalg.norm(rays_d.numpy(), axis=-1)
        # rescaled_distance = rescaled_depth * dist_d
        # selected_rescaled_depth = rescaled_depth.copy()
        # selected_rescaled_depth[rescaled_distance > sphere_intersection_displacement.numpy()] = 1.5

        # saved_depth = rescaled_depth * 5000.0 # if you want to save depth as .png you can use this
        
        # z_depth[z_depth > (max_depth / norm_scale.item()) ] = max_depth / norm_scale.item()
        # z_depth_cam = torch.from_numpy(np.array(Image.fromarray(z_depth).resize(img_size[::-1], Image.NEAREST))).float()
        # depth = norm_scale * z_depth_cam
        # all_depth += [z_depth]
        
        if vis_rgbd: 
            depth_map, _ = visualize_depth_numpy(selected_rescaled_depth.reshape(img_size), [0.0, 1.5])
            rgb_map = (img.numpy() * 255).astype('uint8')
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'rgbd_{index:03d}.png', rgb_map)
        # all_depth += [torch.from_numpy(rescaled_depth).float()]
        # all_depth_save += [torch.from_numpy(saved_depth).float()]

    return torch.stack(all_rgb, dim=0), c2ws, intrinsics

def create_validation_set(src_folder, fraction):
    all_frames = [x.stem for x in sorted(list((src_folder / "color").iterdir()), key=lambda x: int(x.stem))]
    selected_val = [all_frames[i] for i in range(0, len(all_frames), int(1 / fraction))]
    selected_train = [x for x in all_frames if x not in selected_val]
    print(len(selected_train), len(selected_val))
    Path(src_folder / "splits.json").write_text(json.dumps({
        'train': selected_train,
        'test': selected_val
    }))

def idx_to_frame_str(frame_index):
    return f'{frame_index:08d}'

def idx_to_img_filename(frame_index):
    return f'{idx_to_frame_str(frame_index)}.jpg'



def main(args):
    for task in args.tasks:
        if task == 'data_only':
            scene_id = args.scene_name

            dest = Path(args.data_dir) / args.scene_name
            dest.mkdir(exist_ok=True)

            
            output_path = Path(args.output_path) / f"preprocessed" / args.scene_name
            
            rgb_dest = output_path / "images"
            os.makedirs(str(rgb_dest), exist_ok=True)
            scenario_fpath = output_path / "scenario.pt"
            
            scene_objects = dict()
            scene_observers = dict()

            for camName in args.cameras:
                cam_dest = rgb_dest / f"camera_{camName}"
                os.makedirs(str(cam_dest), exist_ok=True)
                
                h, w = args.img_size

                rgbs, poses, intrinsics = read_mate(dest, camName, args.img_size, args.max_depth)
                str_ = f"camera_{camName}"
                if str_ not in scene_observers:
                    scene_observers[str_] = dict(
                        class_name='Camera', n_frames=0, 
                        data=dict(hw=[], intr=[], c2w=[], global_frame_ind=[])
                    )
                for i in range(len(rgbs)):
                    
                    #-------- Process observation groundtruths
                    img = rgbs[i].reshape(args.img_size + [3])
                    img_name = idx_to_img_filename(i)
                    img_path = cam_dest / img_name
                    img = Image.fromarray((img.numpy() * 255).astype(np.uint8)).save(str(img_path))

                    #------------------------------------------------------
                    #------------------     Cameras      ------------------
                    c2w = poses[i]
                    intri = intrinsics[i]
                    scene_observers[str_]['n_frames'] += 1
                    scene_observers[str_]['data']['hw'].append((h, w))
                    scene_observers[str_]['data']['intr'].append(intri.numpy())
                    scene_observers[str_]['data']['c2w'].append(c2w.numpy())
                    scene_observers[str_]['data']['global_frame_ind'].append(i)
                # create_validation_set(save_path, 0.1)
            
            # world_offset = poses[0].numpy().reshape(4, 4)[:3, 3]\
            print(world_offset)
            scene_metas = dict(world_offset=world_offset)
            # scene_metas['dynamic_stats'] = None
            scene_metas['n_frames'] = i + 1

            scenario = dict()
            scenario['scene_id'] = scene_id
            scenario['metas'] = scene_metas
            scenario['objects'] = scene_objects
            scenario['observers'] = scene_observers
            with open(scenario_fpath, 'wb') as f:
                pickle.dump(scenario, f)
                print(f"=> scenario saved to {scenario_fpath}")
        
        elif task == 'omni_only':  
            # do this after data only
            omnidata_normal = OmnidataModel('normal', args.pretrained_models, device="cuda:0")
            omnidata_depth = OmnidataModel('depth', args.pretrained_models, device="cuda:0")

            cameras = ["camera_" + camera for camera in args.cameras]

            output_path = Path(args.output_path) / f"preprocessed" / args.scene_name

            for camera in cameras:
                print(f'================ Process {camera} camera ================')

                img_path = output_path / "images" / camera
                assert img_path.exists(), "Don't exist images file, please do data_only first"

                gen = (i for i in img_path.glob('*.jpg'))
                fnames = sorted(Counter(gen))
                
                for i, img_fname in tqdm(enumerate(fnames)):
                    if args.omin_tasks is not None and Path(args.pretrained_models).exists():
                        for omin_task in args.omin_tasks:
                            out_path = output_path / f"{omin_task}" / camera
                            os.makedirs(str(out_path), exist_ok=True)
        
                            if omin_task == 'normals':
                                prediction = omnidata_normal(img_fname)
                            elif omin_task == 'depths':
                                prediction = omnidata_depth(img_fname)
                            post_prediction_SSF(prediction, img_fname, out_path)
        
        elif task == 'masks_only':
            from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
            from mmseg.core.evaluation import get_palette
            if args.masks_config is None:
                args.masks_config = os.path.join(args.segformer_path, 'local_configs', 'segformer', 'B5', 'segformer.b5.1024x1024.city.160k.py')
            if args.checkpoint is None:
                args.checkpoint = os.path.join(args.segformer_path, 'pretrained', 'segformer.b5.1024x1024.city.160k.pth')

            model = init_segmentor(args.masks_config, args.checkpoint, device=args.device)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="road data preprocessing")
    parser.add_argument("--data_dir", type=str, default='/opt/data/private/chenlu/AutoStudio/AutoStudio/data/carla/',
                    help="")
    parser.add_argument("--scene_name",type=str, default='0',
                       help="")
    parser.add_argument("--img_size", type=int, default=[512, 768], action="append")
    parser.add_argument("--max_depth", type=float, default=100, help="max depth each image can see")
    parser.add_argument("--near_far", type=int, default=[512, 768], action="append")
    parser.add_argument("--cameras", type=str, default=["front_120", "back_100", "leftback_100", "rightback_100", "leftfront_100", "rightfront_100"],
                    help="")
    parser.add_argument("--save_path_name", type=str, default="Carla_SSF")
    # task
    parser.add_argument("--tasks", type=str, default=['omni_only'],
                    choices=[['data_only'], ['omni_only'], ['masks_only'],['depth_only'], ['vis_points']],
                    help="")
    
    # omnidata options 
    parser.add_argument("--omin_tasks", type=str, default=['normals','depths'],
                       choices=[['normal'], ['depth'], ['normal', 'depth']],
                       help="")
    parser.add_argument("--pretrained_models", type=str, default='/opt/data/private/chenlu/AutoStudio/AutoStudio/pretrained_models/pretrained_omnidata',
                       help="")
    parser.add_argument('--patch_size', type=int, default=32, help="")


    parser.add_argument("--output_path", type=str, default= '/opt/data/private/chenlu/AutoStudio/AutoStudio/data/carla',
                       help="Path to store colorized predictions in png")
    
    # masks options
    parser.add_argument('--segformer_path', type=str, default='/opt/data/private/chenlu/AutoStudio/AutoStudio/External/SegFormer')
    parser.add_argument('--masks_config', help='Config file', type=str, default=None)
    parser.add_argument('--masks_checkpoint', help='Checkpoint file', type=str, default=None)
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--palette', default='cityscapes', help='Color palette used for segmentation map')  

    args = parser.parse_args()


    main(args)
    print("Finish Pre-process CARLA Dataset!!!!!!!")
