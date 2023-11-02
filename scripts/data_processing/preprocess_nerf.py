


import os
import imageio
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import List
from collections import Counter
from tqdm import tqdm
from PIL import Image
import sys
sys.path.append(os.path.realpath('./scripts'))
print(sys.path)
from lib_tools.depth_normals.mini_omnidata import OmnidataModel
from utils import post_prediction, visualize_depth

import matplotlib.pyplot as plt
import matplotlib.patches as patches


opengl2waymo = np.array([[0, 0, -1, 0],
                        [-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])

trafo2 = np.array([[-1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])


def main(args):

    # dataset_iter = dataset.as_numpy_iterator()
    # process waymo data only
    if args.tasks == 'data_only':
        SCALE = 1.0
        FILENAME = args.data_dir + '/' + args.scene_name

        data_dir = Path(FILENAME)
        data_dir.mkdir(exist_ok=True)

        output_path = Path(args.output_path) / "preprocessed" / args.scene_name
        os.makedirs(str(output_path), exist_ok=True)

        rgb_dest = output_path / "images"
        os.makedirs(str(rgb_dest), exist_ok=True)

        intrinsic_dest = output_path / "intrinsic"
        os.makedirs(str(intrinsic_dest), exist_ok=True)

        pose_dest = output_path / "poses"
        os.makedirs(str(pose_dest), exist_ok=True)
        
        rgb_dir = data_dir / f"images"
        img_fnames = sorted(os.listdir(rgb_dir))

        cams_meta = np.load(str(data_dir / "cams_meta.npy"))
        poses = cams_meta[:, 0:12].reshape((-1, 3, 4))
        intris = cams_meta[:, 12:21].reshape((-1, 3, 3))
        other_info = cams_meta[:, 21:27]

        for idx, fname  in enumerate(tqdm(img_fnames, desc=f"Loading data({len(img_fnames)})")):
            img = Image.open(rgb_dir / fname )
            # img = torch.from_numpy(np.array(img) / 255.0).float()
            # img = imageio.imread(str(rgb_dir / fname))
            # img = torch.from_numpy(np.array(img) / 255.0).float()

            pose = np.eye(4)
            pose[:3, :4] = poses[idx]
            # pose = poses[idx]
            intri = intris[idx]

            np.savetxt(intrinsic_dest / f"{str(idx).zfill(8)}.txt", intri)
            np.save(intrinsic_dest / f"{str(idx).zfill(8)}.npy", intri)

            np.savetxt(pose_dest / f"{str(idx).zfill(8)}.txt", pose)
            np.save(pose_dest / f"{str(idx).zfill(8)}.npy", pose)
            
            img.save(rgb_dest/f"{str(idx).zfill(8)}.jpg")

        np.save(output_path / "cam_info.npy", other_info)
    
    elif args.tasks == 'omni_only':        
        # do this after data only
        omnidata_normal = OmnidataModel('normal', args.pretrained_models, device="cuda:0")
        omnidata_depth = OmnidataModel('depth', args.pretrained_models, device="cuda:0")

        cameras = ["CAM_" + camera for camera in args.cameras]

        output_path = Path(args.output_path) / f"preprocessed_{args.scene_name}"

        for camera in cameras:
            print(f'================ Process {camera} camera ================')

            img_path = output_path / cameras[0] / "images"
            assert img_path.exists(), "Don't exist images file, please do data_only first"

            gen = (i for i in img_path.glob('*.png'))
            fnames = sorted(Counter(gen))
            
            for i, img_fname in tqdm(enumerate(fnames)):
                if args.omin_tasks is not None and Path(args.pretrained_models).exists():
                    for omin_task in args.omin_tasks:
                        out_vis_path = output_path / camera / f"vis_{omin_task}"
                        os.makedirs(str(out_vis_path), exist_ok=True)
                        out_npy_path = output_path / camera / f"npy_{omin_task}"
                        os.makedirs(str(out_npy_path), exist_ok=True)

                        if omin_task == 'normal':
                            prediction = omnidata_normal(img_fname)
                        elif omin_task == 'depth':
                            prediction = omnidata_depth(img_fname)
                        post_prediction(prediction, img_fname, out_vis_path, out_npy_path)
    else:
        raise KeyError("The Key of Tasks is invalid!!!!")        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name",type=str, default='ngp_fox',
                       help="")
    parser.add_argument("--data_dir", type=str, default='/opt/data/private/chenlu/AutoStudio/AutoStudio/data/synthetic_nerf/ori_data',
                       help="")
    parser.add_argument("--version", type=str, default='',
                       help="")
    parser.add_argument("--verbose", type=bool, default=True,
                       help="")
    
    # task
    parser.add_argument("--tasks", type=str, default='data_only',
                    choices=['data_only', 'omni_only', 'depth_only', 'vis_points'],
                    help="")
    # sky segmentation option

    
    
    # omnidata options 
    parser.add_argument("--omin_tasks", type=str, default=['normal','depth'],
                       choices=[['normal'], ['depth'], ['normal', 'depth']],
                       help="")
    parser.add_argument("--pretrained_models", type=str, default='/opt/data/private/chenlu/AutoStudio/AutoStudio/pretrained_models/pretrained_omnidata',
                       help="")
    parser.add_argument('--patch_size', type=int, default=32, help="")


    parser.add_argument("--output_path", type=str, default= '/opt/data/private/chenlu/AutoStudio/AutoStudio/data/synthetic_nerf/',
                       help="Path to store colorized predictions in png or jpg")

    args = parser.parse_args()

    main(args)