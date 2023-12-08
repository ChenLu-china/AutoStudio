import os
import math
import json
import torch
import imageio
import argparse
import plyfile
import numpy as np
import open3d as o3d

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from typing import Optional
from collections import Counter
from kornia.core import Tensor, stack
from kornia.utils._compat import torch_meshgrid

cam_infos = {
    "front_120": {"fov":120, "x": 2.3, "y": 0, "z": 1.2, "yaw": 0.0},
    "leftfront_100": {"fov":100, "x": 1.1, "y": -1.1, "z": 1.2, "yaw": -45.0},
    "rightfront_100": {"fov":100, "x": 1.1, "y": 1.1, "z": 1.2, "yaw": 45.0},
    "back_100": {"fov":100, "x": -2.3, "y": 0.0, "z": 1.2, "yaw": 180.0},
    "leftback_100": {"fov":100, "x": 1.3, "y": -1.1, "z": 1.2, "yaw": -133.0}, 
    "rightback_100": {"fov":100, "x": 1.3, "y": 1.1, "z": 1.2, "yaw": 133.0}  
}

world_offset = np.array([  1.21562983, -44.79893401, -20.17243147])


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

def depth2array(depth):
    bgr_depth = depth[..., ::-1]
    array = bgr_depth.astype(np.float32)
    normalized_depth = np.dot(array[:,:,:3], [256.0 * 256.0, 256.0, 1.0])
    normalized_depth /= (256.0 * 256.0 * 256.0 - 1.0)
    return normalized_depth # * 1000.0

def to_unreal_matrix(
    to_unreal_location = [0.0, 0.0, 0.0],
    to_unreal_rotation = [-90.0, 0.0, 90.0],
    to_unreal_scale = [-1.0, 1.0, 1.0]):

    return get_matrix(to_unreal_location, to_unreal_rotation, to_unreal_scale)

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
    else:
        translate = -world_offset
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
        
        # opencv_to_opengl = np.eye(4)
        # opencv_to_opengl[:3 ,:3] = np.array(
        #     [[1, 0, 0],
        #     [0, -1, 0],
        #     [0, 0, -1]])
        # c2w = c2w @ opencv_to_opengl

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
    # if vis_normCamera:
    #     for index in sample_indices:
    #         frame = {f"{index}.png":{"K": intrinsics_rz[index].tolist(), "C2W":norm_poses[index].tolist(), "img_size":img_size}}
    #         camera_norm_dict.update(frame)
    #     saveFile = root_dir / "camera_dict_norm.json"
    #     with open(saveFile, 'w') as f:
    #         json.dump(camera_norm_dict, f, ensure_ascii=False, indent=4)
    #     f.close()

    return torch.from_numpy(norm_poses).float(), torch.stack(intrinsics_rz).float() # , translate, scale

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

def calculate_pc_distance(rays_o, rays_d, k, normalized_depth, rgb, far, scale, max_depth=70.0):
    # H, W = img_size
    far = far * scale

    rgb = rgb.reshape(-1, 3)
    z_depth = normalized_depth.reshape(-1) * far
    max_depth_indexes = np.where(z_depth <= max_depth)
    z_depth = z_depth[max_depth_indexes]
    rays_o = rays_o[max_depth_indexes]
    rays_d = rays_d[max_depth_indexes]
    rgb = rgb[max_depth_indexes]

    d_dist = np.linalg.norm(rays_d, axis=-1)
    distance = z_depth * d_dist

    viewdirs = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    pcd = rays_o + distance[:, np.newaxis] * viewdirs
    return pcd, rgb


def read_mate(root_dir, cam_name, img_size, max_depth, show_pc:bool = None, vis_rgbd:bool = False):
    
    color_dir = root_dir  / f"color_{cam_name}"
    depth_dir = root_dir  / f"depth_{cam_name}"
    pred_depth_dir = root_dir / f"pred_depth_{cam_name}"
    pose_dir  = root_dir  / f"pose"
    
    num_img = len(os.listdir(color_dir))

    
    c2ws, intrinsics = norm_poses(root_dir, cam_name, num_img, max_depth=max_depth, img_size=img_size)
    # norm_scale = s2n[0, 0]
    
    max_show = 1
    all_pcd, cam_pcl_color, pred_all_pcd, pred_cam_pcl_color = [], [], [], []
    frustums = []
    all_rgb, all_depth, all_depth_save = [], [], []
    for index in tqdm(np.arange(0, max_show, 1), desc=f"Loading data ({num_img})"):

        img = Image.open(color_dir / f"{index}.png")
        img = torch.from_numpy(np.array(img.resize(img_size[::-1], Image.LANCZOS)) / 255.0).float()
        all_rgb += [img]

        c2w = c2ws[index]
        intrinsic = intrinsics[index]

        directions = get_ray_directions_use_intrinsics(img_size[0], img_size[1], intrinsic.numpy())
        rays_o, rays_d = get_rays(directions, c2w)

        z_depth = Image.open(depth_dir / f'{index}.png')  # z-buffer depth
        z_depth = depth2array(np.array(z_depth, dtype=np.float32))
        
        pred_z_depth = np.load(pred_depth_dir / f"{index:08d}.npz")["arr_0"]

        if show_pc:
            if index < max_show:
                pcd, pcd_color = calculate_pc_distance(rays_o.numpy(), rays_d.numpy(), intrinsic, z_depth, img.numpy(), 1000.0, scale=1.0)
                pred_pcd, pred_color = calculate_pc_distance(rays_o.numpy(), rays_d.numpy(), intrinsic, pred_z_depth, img.numpy(), 120.0, scale=1.0)
                # pcd = calculate_pc_zbuff(c2w, intrinsic, z_depth, scale)
                # frustums.append(get_camera_frustum(img_size, intrinsic, c2w, frustum_length=scale))
                all_pcd.append(pcd)
                cam_pcl_color.append(pcd_color)

                pred_all_pcd.append(pred_pcd)
                pred_cam_pcl_color.append(pred_color)
              
                # all_points = all_points
                # show_pc(all_points, cameras)
        
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
        
        # if vis_rgbd: 
        #     depth_map, _ = visualize_depth_numpy(selected_rescaled_depth.reshape(img_size), [0.0, 1.5])
        #     rgb_map = (img.numpy() * 255).astype('uint8')
        #     rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        #     imageio.imwrite(f'rgbd_{index:03d}.png', rgb_map)
        # all_depth += [torch.from_numpy(rescaled_depth).float()]
        # all_depth_save += [torch.from_numpy(saved_depth).float()]
    all_pcd = np.concatenate(all_pcd, 0).reshape(-1, 3)
    cam_pcl_color = np.concatenate(cam_pcl_color, 0).reshape(-1, 3)

    pred_all_pcd = np.concatenate(pred_all_pcd, 0).reshape(-1, 3)
    pred_cam_pcl_color = np.concatenate(pred_cam_pcl_color, 0).reshape(-1, 3)
    
    return all_pcd, cam_pcl_color, pred_all_pcd, pred_cam_pcl_color

def export_pcl_ply(pcl: np.ndarray, pcl_color: np.ndarray = None, filepath: str = ...):
    """
    pcl_color: if provided, should be uint8_t
    """
    num_pts = pcl.shape[0]
    if pcl_color is not None:
        verts_tuple = np.zeros((num_pts,), dtype=[(
            "x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        data = [tuple(p1.tolist() + p2.tolist()) for p1, p2 in zip(pcl, pcl_color)]
        verts_tuple[:] = data[:]
    else:
        verts_tuple = np.zeros((num_pts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        data = [tuple(p.tolist()) for p in pcl]
        verts_tuple[:] = data[:]

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    ply_data = plyfile.PlyData([el_verts])
    print(f"=> Saving pointclouds to {str(filepath)}")
    ply_data.write(filepath)

def main(args):
    
    f_pcd, f_pcd_color, f_pred_pcd, f_pred_pcd_color = [], [], [], []
    select_scene_ids = args.scene_name
    output_path = Path(args.output_path) / f"preprocessed"
    for scene_i, scene_id in enumerate(tqdm(select_scene_ids, f'Extracting masks ...')):
        dest = Path(args.data_dir) / scene_id
        dest.mkdir(exist_ok=True)
        

        for camName in args.cameras:

            pcd, pcd_color, pred_pcd, pred_pcd_color = read_mate(dest, camName, args.img_size, args.max_depth, show_pc=args.gather_cam_pcl)
            f_pcd.append(pcd)
            f_pcd_color.append(pcd_color)
            f_pred_pcd.append(pred_pcd)
            f_pred_pcd_color.append(pred_pcd_color)
    
    pcl_filepath = os.path.join(output_path, f"full_pointcloud.ply")
    f_pcd = np.concatenate(f_pcd, 0).reshape(-1, 3)
    f_pcd_color = np.concatenate(f_pcd_color, 0).reshape(-1, 3)
    f_pcd_color = (f_pcd_color * 255.0).astype(np.uint8)
    export_pcl_ply(f_pcd, f_pcd_color, filepath=pcl_filepath)

    pcl_filepath = os.path.join(output_path, f"pred_full_pointcloud.ply")
    f_pred_pcd = np.concatenate(f_pred_pcd, 0).reshape(-1, 3)
    f_pred_pcd_color = np.concatenate(f_pred_pcd_color, 0).reshape(-1, 3)
    f_pred_pcd_color = (f_pred_pcd_color * 255.0).astype(np.uint8)
    export_pcl_ply(f_pred_pcd, f_pred_pcd_color, filepath=pcl_filepath)
    





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Carla Dataset PointCloud Extraction")
    
    parser.add_argument("--data_dir", type=str, default='/opt/data/private/chenlu/AutoStudio/AutoStudio/data/carla/',
                    help="")
    parser.add_argument("--scene_name",type=int, default=['0'],
                       help="")
    parser.add_argument("--cameras", type=str, default=["front_120", "leftfront_100"],
                    help="")
    # parser.add_argument("--cameras", type=str, default=["front_120", "back_100", "leftback_100", "rightback_100", "leftfront_100", "rightfront_100"],
    #                 help="")
    parser.add_argument("--img_size", type=int, default=[512, 768], action="append")
    parser.add_argument("--max_depth", type=float, default=100, help="max depth each image can see")

    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--stop_frame', type=int, default=None)

    parser.add_argument("--gather_cam_pcl", type=bool, default=True, 
                        help="If set, will produce camera point clouds (colored point clouds projected using camera-rendered depth).")
    

    parser.add_argument("--output_path", type=str, default= '/opt/data/private/chenlu/AutoStudio/AutoStudio/data/carla',
                       help="Path to store colorized predictions in png")

    main(parser.parse_args())