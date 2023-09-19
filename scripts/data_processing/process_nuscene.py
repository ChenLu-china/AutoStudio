import os
import cv2
import argparse
from util import *
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from typing import List
import matplotlib.pyplot as plt

from lib_tools import Omnidata
from . import post_prediction

def main(args):
    nuses = NuScenes(version=args.version, 
                     dataroot=args.data_dir, 
                     verbose=args.version)
    
    cameras = ["CAM_" + camera for camera in args.cameras]

    # get samples for scene
    samples = [samp for samp in nuses.sample if nuses.get("scene", samp["scene_token"])['name'] == str(args.scene_name)]
    # get sweeps for scene 
    samples = [samp for samp in nuses.sample if nuses.get("scene", samp["scene_token"])['name'] == str(args.scene_name)]
    samples.sort(key=lambda x: (x["scene_token"], x["timestamp"]))

    # get image filenames and camera data

    image_filenames = []
    mask_filenames = []
    mask_dir = []
    intrinsics = []
    poses = []

    data_dir = Path(args.data_dir)
    output_path = Path(args.output_path) / f"preprocessed_{args.version}_{args.scene_name}"
    os.makedirs(str(output_path), exist_ok=True)

    omnidata_normal = Omnidata('normal', args.pretrained_models)
    omnidata_depth = Omnidata('depth', args.pretrained_models)

    for samp in samples:
        for camera in cameras:
            camera_data = nuses.get("sample_data", samp["data"][camera])
            calibrated_sensor_data = nuses.get("calibrated_sensor", camera_data["calibrated_sensor_token"])
            ego_pose_data = nuses.get("ego_pose", camera_data["ego_pose_token"])
            ego_pose = rotation_translation_to_pose_np(ego_pose_data["rotation"], 
                                                       ego_pose_data["translation"])

            cam_pose = rotation_translation_to_pose_np(calibrated_sensor_data["rotation"], 
                                                       calibrated_sensor_data["translation"])
            pose = ego_pose @ cam_pose

            pose = nusc2opecv @ pose

            img_fname = data_dir / camera_data["filename"]
            
            out_img_fname = output_path / camera /f"images"
            os.makedirs(str(out_img_fname), exist_ok=True)

            img = cv2.imread(str(img_fname))
            plt.imsave(out_img_fname / f"{img_fname.stem}.png", img)

            out_intrinsic_path = output_path / camera / f"intrinsics"
            os.makedirs(str(out_intrinsic_path), exist_ok=True)
            np.savetxt(out_intrinsic_path / f'{img_fname.stem}.txt', calibrated_sensor_data["camera_intrinsic"])

            out_poses_path = output_path / camera / f"poses"
            os.makedirs(str(out_poses_path), exist_ok=True)
            np.savetxt(out_poses_path / f'{img_fname.stem}.txt', pose)  #opencv extrinsic

            image_filenames.append(data_dir / camera_data["filename"])
            intrinsics.append(calibrated_sensor_data["camera_intrinsic"])
            poses.append(pose)

            if args.omin_tasks is not None and Path(args.pretrained_models).exists():
                for omin_task in args.omin_tasks:
                    out_vis_path = output_path / camera /f"vis_{omin_task}"
                    os.makedirs(str(out_vis_path), exist_ok=True)
                    out_npy_path = output_path / camera /f"npy_{omin_task}"
                    os.makedirs(str(out_npy_path), exist_ok=True)

                    if omin_task == 'normal':
                        prediction = omnidata_normal(img_fname)
                    elif omin_task == 'depth':
                        prediction = omnidata_depth(img_fname)
                    post_prediction(prediction, img_fname, out_vis_path, out_npy_path)


    # for i, fname in enumerate(image_filenames):
    #     out_intrinsic_path = output_path / f"intrinsics"

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name",type=str, default='scene-0916',
                       help="")
    parser.add_argument("--data_dir", type=str, default='/opt/data/private/chenlu/AutoStudio/data/nuscenes/v1.0-mini',
                       help="")
    parser.add_argument("--version", type=str, default='v1.0-mini',
                       choices=["v1.0-mini", "v1.0-trainval"],
                       help="")
    parser.add_argument("--cameras", type=List, default=["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "BACK_LEFT", "BACK_RIGHT"],
                       help="")
    parser.add_argument("--verbose", type=bool, default=True,
                       help="")
    
    # sky segmentation option

    
    
    # omnidata options 
    parser.add_argument("--omin_tasks", type=str, default=['normal','depth'],
                       choices=[['normal'], ['depth'], ['normal', 'depth']],
                       help="")
    parser.add_argument("--pretrained_models", type=str, default='/opt/data/private/chenlu/AutoStudio/pretrained_models/pretrained_omnidata',
                       help="")
    parser.add_argument("--output_path", type=str, default= '/opt/data/private/chenlu/AutoStudio/data/nuscenes/',
                       help="Path to store colorized predictions in png")

    args = parser.parse_args()

    main(args)
    pass