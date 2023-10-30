"""
Run this python file will do follow works:
"""

# Copyright 2023 Lu Chen
# reference https://github.com/FelTris/durf/blob/main/notebooks/waymo_data.ipynb by FelTris
# and https://github.com/PJLab-ADG/neuralsim/blob/19b5b33113d09676bc72dca7c94b640c73d99710/dataio/autonomous_driving/waymo/preprocess.py by ventusff

import os 
import tensorflow as tf
import argparse
import numpy as np
from pathlib import Path
from typing import List
from collections import Counter
from tqdm import tqdm

import sys
sys.path.append(os.path.realpath('./scripts'))
print(sys.path)
from lib_tools.depth_normals.mini_omnidata import OmnidataModel
from utils import post_prediction, visualize_depth

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

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
        FILENAME = args.data_dir + '/' + args.version

        dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')    
        frames = []
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))  
            frames.append(frame)
        
        data_dir = Path(args.data_dir)
        output_path = Path(args.output_path) / f"preprocessed_{args.scene_name}"
        os.makedirs(str(output_path), exist_ok=True)

        # process waymo images only
        for i, elem in tqdm(enumerate(frames)):
            for j, img in tqdm(enumerate(elem.images)):
                # print(img.name)
                
                cam_name = open_dataset.CameraName.Name.Name(img.name)
                # print(cam_name)
                
                # process images
                out_img_fname = output_path / f"CAM_{cam_name}" / "images"
                os.makedirs(str(out_img_fname), exist_ok=True)
                
                image = np.array(tf.image.decode_jpeg(img.image))
                plt.imsave(out_img_fname / f"{str(i).zfill(8)}.png", image)
                
                # process intrinsics
                out_intrinsic_path = output_path / f"CAM_{cam_name}" / "intrinsic"
                os.makedirs(str(out_intrinsic_path), exist_ok=True)

                w = elem.context.camera_calibrations[img.name-1].width
                h = elem.context.camera_calibrations[img.name-1].height
                focal = elem.context.camera_calibrations[img.name-1].intrinsic[0]
                cx = elem.context.camera_calibrations[img.name-1].intrinsic[2]
                cy = elem.context.camera_calibrations[img.name-1].intrinsic[3]

                intrinsic = np.eye(3)
                intrinsic[0, 0] = intrinsic[1, 1] = focal
                intrinsic[0, 2] = cx
                intrinsic[1, 2] = cy

                # intrinsic_ = frame.context.camera_calibrations[img.name-1].intrinsic

                np.savetxt(out_intrinsic_path / f"{str(i).zfill(8)}.txt", intrinsic)
                np.save(out_intrinsic_path / f"{str(i).zfill(8)}.npy", intrinsic)

                # process poses
                out_poses_path = output_path / f"CAM_{cam_name}" / "poses"
                os.makedirs(str(out_poses_path), exist_ok=True)

                v2w = np.asarray(elem.pose.transform).reshape(4,4)
                c2v = np.asarray(elem.context.camera_calibrations[img.name-1].extrinsic.transform).reshape(4,4)
                
                pose = np.matmul(v2w, c2v)
                pose = np.matmul(pose, opengl2waymo)  # under opengl
                # pose = np.matmul(trafo2, pose)

                np.savetxt(out_poses_path / f"{str(i).zfill(8)}.txt", pose)  #opencv extrinsic
                np.save(out_poses_path / f"{str(i).zfill(8)}.npy", pose)  #opencv extrinsic
                
    elif args.tasks == 'depth_only':
        SCALE = 1.0
        FILENAME = args.data_dir + '/' + args.version

        dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')    
        frames = []
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))  
            frames.append(frame)
        
        
        data_dir = Path(args.data_dir)
        output_path = Path(args.output_path) / f"preprocessed_{args.scene_name}"
        os.makedirs(str(output_path), exist_ok=True)

        for i, elem in enumerate(frames):
            for j, img in enumerate(elem.images):

                cam_name = open_dataset.CameraName.Name.Name(img.name)
                # print(cam_name)
                
                # process depth
                out_depth_path = output_path / f"CAM_{cam_name}" / "lidar_depth"
                os.makedirs(str(out_depth_path), exist_ok=True)


                (range_images, camera_projections, _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(elem)
                
                points, cp_points = frame_utils.convert_range_image_to_point_cloud(elem,
                                                                                   range_images,
                                                                                   camera_projections,
                                                                                   range_image_top_pose)
                # 3d points in VEHICLE frame.
                points_all = np.concatenate(points, axis=0)
                # camera projection corresponding to each point.
                cp_points_all = np.concatenate(cp_points, axis=0)
                cp_points_all[:,1] = cp_points_all[:,1] / SCALE
                cp_points_all[:,2] = cp_points_all[:,2] / SCALE
                cp_points_all[:,4] = cp_points_all[:,4] / SCALE
                cp_points_all[:,5] = cp_points_all[:,5] / SCALE
                
                cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
                cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

                # The distance between lidar points and vehicle frame origin.
                points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
                cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)
                                
                mask = tf.equal(cp_points_all_tensor[..., 0], img.name)
                overlap = tf.equal(cp_points_all_tensor[..., 3], img.name)

                cp_points_all_tensor_mask = tf.cast(tf.gather_nd(cp_points_all_tensor,
                                                            tf.where(mask)), dtype=tf.float32)
                cp_points_all_tensor_overlap = tf.cast(tf.gather_nd(cp_points_all_tensor,
                                                            tf.where(overlap)), dtype=tf.float32)
                
                points_all_tensor_mask = tf.gather_nd(points_all_tensor, tf.where(mask))
                points_all_tensor_overlap = tf.gather_nd(points_all_tensor, tf.where(overlap))

                projected_points_all_from_raw_data = tf.concat([cp_points_all_tensor_mask[..., 1:3],
                                                                points_all_tensor_mask], axis=-1).numpy()
                projected_point_overlap = tf.concat([cp_points_all_tensor_overlap[..., 4:6],
                                                                points_all_tensor_overlap], axis=-1).numpy()

                if projected_point_overlap.shape[0] != 0:
                    projected_points_all_from_raw_data = np.concatenate([projected_points_all_from_raw_data,
                                                                       projected_point_overlap])

                w = int(elem.context.camera_calibrations[img.name-1].width / SCALE)
                h = int(elem.context.camera_calibrations[img.name-1].height / SCALE)
                resolution = (h, w)

                depth = np.zeros(resolution)
                for pts in projected_points_all_from_raw_data:
                    if pts[0] < resolution[1] and  pts[1] < resolution[0]:
                        if pts[2] < depth[int(pts[1]), int(pts[0])] or depth[int(pts[1]), int(pts[0])] == 0:
                            depth[int(pts[1]), int(pts[0])] = pts[2]
                np.save(out_depth_path / f"{str(i).zfill(8)}.npz", depth)
                # img_d = visualize_depth(depth)
                # img_d.save("lidar_depth.png")
    
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
    
    elif args.tasks == 'vis_points':
        output_path = Path(args.output_path) / f"preprocessed_{args.scene_name}"
        FILENAME = args.data_dir + '/' + args.version

        dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')    
        frames = []
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))  
            frames.append(frame)
        
        import open3d as o3d
        (range_images, camera_projections, _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frames[0])
        # points are given in vehicle coordinates
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frames[0],
            range_images,
            camera_projections,
            range_image_top_pose)
        points_all = np.concatenate(points, axis=0)
        points_hom = np.ones((points_all.shape[0], 4))
        points_hom[:,:3] = points_all

        # this transforms from vehicle to world coordinates
        v2w = np.asarray(frames[0].pose.transform).reshape(4,4)

        # opengl2waymo = np.array([[0, 0, -1, 0],
        #                         [-1, 0, 0, 0],
        #                         [0, 1, 0, 0],
        #                         [0, 0, 0, 1]])

        # transform points to world coordinates
        points_w = np.matmul(points_hom, v2w.T)
        points_wgl = np.matmul(points_w, opengl2waymo)

        test = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_wgl[:,:3]))
        o3d.io.write_point_cloud(str(output_path) + '/' + "test0.ply", test)
    else:
        raise KeyError("The Key of Tasks is invalid!!!!")        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name",type=str, default='segment-10061305430875486848_1080_000_1100_000_with_camera_labels',
                       help="")
    parser.add_argument("--data_dir", type=str, default='/opt/data/private/chenlu/AutoStudio/data/waymo',
                       help="")
    parser.add_argument("--version", type=str, default='segment-10061305430875486848_1080_000_1100_000_with_camera_labels.tfrecord',
                       help="")
    parser.add_argument("--cameras", type=List, default=["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"],
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
    parser.add_argument("--pretrained_models", type=str, default='/opt/data/private/chenlu/AutoStudio/pretrained_models/pretrained_omnidata',
                       help="")
    parser.add_argument('--patch_size', type=int, default=32, help="")


    parser.add_argument("--output_path", type=str, default= '/opt/data/private/chenlu/AutoStudio/data/waymo/',
                       help="Path to store colorized predictions in png")

    args = parser.parse_args()

    main(args)
    