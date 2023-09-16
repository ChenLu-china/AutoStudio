import os 
import cv2
import math
import json
import torch
import imageio
import numpy as np

from .ray_utils import *
# from .util import *
from tqdm import tqdm
from PIL import Image
from typing import List
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms as T

class CarlaDataset(Dataset):
    """
        We have already preprocessed camera poses into a unit space
    """
    def __init__(self,
                datadir: str,
                split: str = 'train',
                is_stack: bool = False,
                N_vis: int = -1,
                img_sz: List = [512, 768],
                downsample: int = -1,
                use_depth: bool = True
                ) -> None:
        
        self.root_dir = Path(datadir)
        self.N_vis = N_vis
        self.split = split
        self.img_wh = [img_sz[1], img_sz[0]]
        self.img_sz = img_sz
        self.is_stack = is_stack
        self.use_depth = use_depth
        self.downsample = downsample
        self.define_transforms()
        
        self.scene_bbox = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # do not use this 
        
        self.all_frame_names = sorted([x.stem for x in (self.root_dir / "color").iterdir() if x.name.endswith('.png')], key=lambda y: int(y) if y.isnumeric() else y)
        if (self.root_dir/ "splits.json").exists():
            split_json = json.loads((self.root_dir / "splits.json").read_text())
            self.train_indices = [self.all_frame_names.index(f'{x}') for x in split_json['train']]
            if self.split == 'test':
                self.test_indices = [self.all_frame_names.index(f'{x}') for x in split_json['test']]
                self.test_indices = self.test_indices[::self.downsample]
        self.train_indices = self.train_indices[::self.downsample]    
        
        self.scene_scale = json.loads((self.root_dir / "camera_dict_norm.json").read_text())["Norm_Scale"]
        
        if self.split == 'train':
            self.read_meta(self.train_indices, use_depth)
        else:
            self.read_meta(self.test_indices, use_depth)

        self.white_bg = False
        self.near_far = [0.0, 1.5]
        
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.raidus = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample = downsample

    def read_meta(self, indices, use_depth:bool=True):
        
        poses, intrinsics, all_rays, all_rgbs, all_depth = [], [], [], [], [] # H, W
        depth_masks = []

        img_h, img_w = np.array(Image.open(self.root_dir / "color" / f"{self.all_frame_names[0]}.png")).shape[:2]
        for idx, sample_index in enumerate(tqdm(indices, desc=f'Loading data {self.split} ({len(indices)})')):

            # c2ws, intrinsics_rz = self.norm_pose(self.img_sz) # re-center camera poses to (0, 0, 0)
            img = Image.open(self.root_dir / "color" / f"{self.all_frame_names[sample_index]}.png")
            img = torch.from_numpy(np.array(img.resize(self.img_sz[::-1], Image.LANCZOS)) / 255.0).float()
            img = img.view(-1, 3)
            all_rgbs += [img]
            
            intrinsic = np.loadtxt(self.root_dir / "intrinsic" / f"{self.all_frame_names[sample_index]}.txt")
            intrinsic = torch.from_numpy(np.diag([self.img_sz[1] / img_w , self.img_sz[0] / img_h, 1]) @ intrinsic).float()
            intrinsics += [intrinsic]

            c2w = np.loadtxt(self.root_dir / "pose"/ f"{self.all_frame_names[sample_index]}.txt")
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w] 

            directions = get_ray_directions_use_intrinsics(self.img_sz[0], self.img_sz[1], intrinsic.numpy())
            rays_o, rays_d = get_rays(directions, c2w)
            sphere_intersection = rays_intersect_sphere(rays_o, rays_d, r=1) # calulate far t
            all_rays += [torch.cat([rays_o, rays_d, 
                                     0.0 * torch.ones_like(rays_o[:, :1]), 
                                     sphere_intersection[:, None]], 1)] # len(h*w, 8)
            
            if use_depth:
                raw_depth = np.load(self.root_dir / "depth_npy" / f"{self.all_frame_names[sample_index]}.npy") # rescaled z-buffer depth
                raw_depth = torch.from_numpy(np.array(Image.fromarray(raw_depth).resize(self.img_sz[::-1], Image.NEAREST)))
                raw_depth = raw_depth.view(-1)
                rays_d_dist = torch.norm(rays_d, dim=-1)
                raw_dist = raw_depth * rays_d_dist
                depth_mask = raw_dist < sphere_intersection
                
                all_depth += [raw_depth]
                depth_masks += [depth_mask]

                # img[depth_mask == 0] = torch.tensor([1., 1., 1.])
                # Image.fromarray((img.numpy().reshape(self.img_sz + [3]) * 255).astype(np.uint8)).save("./masked_rgb.png")
                # raw_depth = 
        
        self.poses = torch.stack(poses)
        self.intrinsics = torch.stack(intrinsics)

        if not self.is_stack:
            self.all_rays = torch.cat(all_rays, 0)
            self.all_rgbs = torch.cat(all_rgbs, 0)
            self.all_depth = torch.cat(all_depth, 0)
            self.depth_masks = torch.cat(depth_masks, 0)
        else:
            self.all_rays = torch.stack(all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(all_rgbs, 0).reshape(-1,*self.img_sz[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            self.all_depth = torch.stack(all_depth, 0).reshape(-1, *self.img_sz[::-1])
            self.depth_masks = torch.stack(depth_masks, 0).reshape(-1, *self.img_sz[::-1])

    def define_transforms(self):
        self.transform = T.ToTensor()   

    def world2ndc(self,points,lindisp=None):
            device = points.device
            return (points - self.center.to(device)) / self.radius.to(device)

    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            # mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img}
        return sample

if __name__ == '__main__':
    pass
