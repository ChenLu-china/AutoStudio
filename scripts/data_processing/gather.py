"""
Gather different trajectories of carla to one dataset format

"""

def idx_to_frame_str(frame_index):
    return f'{frame_index:08d}'

def idx_to_img_filename(frame_index):
    return f'{idx_to_frame_str(frame_index)}'

if __name__ == "__main__":
    import os
    import imageio
    import pickle
    import shutil
    import numpy as np
    from glob import glob
    from tqdm import tqdm
    from pathlib import Path
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # Custom configs
    parser.add_argument('--data_root', type=str, default='/opt/data/private/chenlu/AutoStudio/AutoStudio/data/carla/preprocessed')
    parser.add_argument('--seq_list', type=str, default=None, help='specify --seq_list if you want to limit the list of seqs')
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--ignore_existing', action='store_true')
    parser.add_argument('--no_compress', action='store_true')
    parser.add_argument('--rgb_dirname', type=str, default="images")
    parser.add_argument('--depth_dirname', type=str, default=None)
    parser.add_argument('--normal_dirname', type=str, default=None)
    parser.add_argument('--mask_dirname', type=str, default=None)
    parser.add_argument('--lidar_dirname', type=str, default="lidars")
    parser.add_argument('--out_dirname', type=str, default="full")

    # Algorithm configs
    parser.add_argument('--segformer_path', type=str, default='/opt/data/private/chenlu/AutoStudio/AutoStudio/External/SegFormer')
    parser.add_argument('--config', help='Config file', type=str, default=None)
    parser.add_argument('--checkpoint', help='Checkpoint file', type=str, default="/opt/data/private/chenlu/AutoStudio/AutoStudio/pretrained_models/pretrained_segformer/segformer.b5.1024x1024.city.160k.pth")
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--palette', default='cityscapes', help='Color palette used for segmentation map')
    
    args = parser.parse_args()

    if args.seq_list is not None:
        with open(args.seq_list, 'r') as f:
            seq_list = f.read().splitlines()
        select_scene_ids = [s.split(',')[0].rstrip(".tfrecord") for s in seq_list]
    else:
        select_scene_ids = list(sorted(glob(os.path.join(args.data_root, "*", "scenario.pt"))))
        if len(select_scene_ids) <=0:
            select_scene_ids = ['0']
        else:
            select_scene_ids = [os.path.split(os.path.dirname(s))[-1] for s in select_scene_ids]
    
    scene_objects = dict()
    scene_observers = dict()

    out_dir = os.path.join(args.data_root, args.out_dirname)
    os.makedirs(out_dir, exist_ok=True)
    out_scenario_fpath = os.path.join(out_dir , "scenario.pt")

    global_num = 0

    for scene_i, scene_id in enumerate(tqdm(select_scene_ids, f'Moving ...')):
        obs_id_list = sorted(os.listdir(os.path.join(args.data_root, scene_id, args.rgb_dirname)))
        lidar_id_list = sorted(os.listdir(os.path.join(args.data_root, scene_id, args.lidar_dirname)))
        scenario_fpath = os.path.join(args.data_root, scene_id, "scenario.pt")
        if os.path.exists(scenario_fpath):
            with open(scenario_fpath, 'rb') as f:
                scenario = pickle.load(f)
            part_scene_observers = scenario['observers']
        for obs_i, obs_id in enumerate(tqdm(obs_id_list, f'scene [{scene_i}/{len(select_scene_ids)}]')):
            
            if obs_id not in scene_observers:
                scene_observers[obs_id] = dict(
                    class_name='Camera', n_frames=0, 
                    data=dict(hw=[], intr=[], c2w=[], global_frame_ind=[])
                )

            lidar_id = lidar_id_list[obs_i]
            if lidar_id not in scene_observers:
                scene_observers[lidar_id] = dict(
                    class_name='RaysLidar', n_frames=0, 
                        data=dict(l2v=[], l2w=[], global_frame_ind=[])
                )

            if args.rgb_dirname is not None: 
                img_dir = os.path.join(args.data_root, scene_id, args.rgb_dirname, obs_id)
                out_img_dir = os.path.join(out_dir, args.rgb_dirname, obs_id)
                os.makedirs(out_img_dir, exist_ok=True)
            
            if args.normal_dirname is not None: 
                normal_dir = os.path.join(args.data_root, scene_id, args.normal_dirname, obs_id)
                out_normal_dir = os.path.join(out_dir, args.normal_dirname, obs_id)
                os.makedirs(out_normal_dir, exist_ok=True)

            if args.depth_dirname is not None:
                depth_dir = os.path.join(args.data_root, scene_id, args.depth_dirname, obs_id)
                out_depth_dir = os.path.join(out_dir, args.depth_dirname, obs_id)
                os.makedirs(out_depth_dir, exist_ok=True)

            if args.mask_dirname is not None: 
                mask_dir = os.path.join(args.data_root, scene_id, args.mask_dirname, obs_id)
                out_mask_dir = os.path.join(out_dir, args.mask_dirname, obs_id)
                os.makedirs(out_mask_dir,exist_ok=True)
            

            if args.lidar_dirname is not None: 
                lidar_dir = os.path.join(args.data_root, scene_id, args.lidar_dirname, lidar_id)
                out_lidar_dir = os.path.join(out_dir, args.lidar_dirname, lidar_id)
                os.makedirs(out_lidar_dir, exist_ok=True)
                
            flist = sorted(glob(os.path.join(img_dir, '*.jpg')))
            for f_i, fpath in enumerate(tqdm(flist, f'scene[{scene_i}][{obs_id}]')):
                fformer = Path(fpath).parent
                fbase = Path(fpath).stem
                fbase_new = idx_to_img_filename(scene_observers[obs_id]['n_frames'] + f_i)
                
                if args.rgb_dirname is not None:
                    dst_img = os.path.join(out_img_dir, f"{fbase_new}.jpg")
                    shutil.copyfile(fpath, dst_img)
                
                if args.normal_dirname is not None: 
                    src_normal = os.path.join(normal_dir, f"{fbase}.npz")
                    src_normal_img = os.path.join(normal_dir, f"{fbase}.jpg")
                    dst_normal = os.path.join(out_normal_dir, f"{fbase_new}.npz")
                    dst_normal_img = os.path.join(out_normal_dir, f"{fbase_new}.jpg")
                    shutil.copyfile(src_normal, dst_normal)
                    shutil.copyfile(src_normal_img, dst_normal_img)
                    
                if args.depth_dirname is not None:
                    src_depth = os.path.join(depth_dir, f"{fbase}.npz")
                    src_depth_img = os.path.join(depth_dir, f"{fbase}.jpg")
                    dst_depth = os.path.join(out_depth_dir, f"{fbase_new}.npz")
                    dst_depth_img = os.path.join(out_depth_dir, f"{fbase_new}.jpg")
                    shutil.copyfile(src_depth, dst_depth)
                    shutil.copyfile(src_depth_img, dst_depth_img)


                if args.mask_dirname is not None: 
                    src_mask = os.path.join(mask_dir, f"{fbase}.npz")
                    src_mask_img = os.path.join(mask_dir, f"{fbase}.jpg")
                    dst_mask = os.path.join(out_mask_dir, f"{fbase_new}.npz")
                    dst_mask_img = os.path.join(out_mask_dir, f"{fbase_new}.jpg")   
                    shutil.copyfile(src_mask, dst_mask)
                    shutil.copyfile(src_mask_img, dst_mask_img)

                if args.lidar_dirname is not None: 
                    src_lidar = os.path.join(lidar_dir, f"{fbase}.npz")
                    dst_lidar = os.path.join(out_lidar_dir, f"{fbase_new}.npz")
                    shutil.copyfile(src_lidar, dst_lidar)
                
                #---- move outputs
                if os.path.exists(scenario_fpath):
                    scene_observers[obs_id]['data']['hw'].append(part_scene_observers[obs_id]['data']['hw'][f_i])
                    scene_observers[obs_id]['data']['intr'].append(part_scene_observers[obs_id]['data']['intr'][f_i])
                    scene_observers[obs_id]['data']['c2w'].append(part_scene_observers[obs_id]['data']['c2w'][f_i])
                    scene_observers[obs_id]['data']['global_frame_ind'].append(scene_observers[obs_id]['n_frames'] + f_i)
                    scene_observers[lidar_id]['data']['l2v'].append(part_scene_observers[lidar_id]['data']['l2v'][f_i])
                    scene_observers[lidar_id]['data']['l2w'].append(part_scene_observers[lidar_id]['data']['l2w'][f_i])
                    scene_observers[lidar_id]['data']['global_frame_ind'].append(scene_observers[lidar_id]['n_frames'] + f_i)

            if os.path.exists(scenario_fpath):
                scene_observers[obs_id]['n_frames'] += part_scene_observers[obs_id]['n_frames']
                scene_observers[lidar_id]['n_frames'] += part_scene_observers[lidar_id]['n_frames']
                global_num += part_scene_observers[obs_id]['n_frames']
    

    world_offset = scenario['metas']['world_offset']
    scene_id = "full"
    scenario = dict()
    scene_metas = dict(world_offset=world_offset)
    
    scene_metas['n_frames'] = global_num // 6
    scenario['scene_id'] = scene_id
    scenario['metas'] = scene_metas
    scenario['objects'] = scene_objects
    scenario['observers'] = scene_observers
    
    with open(out_scenario_fpath, 'wb') as f:
                pickle.dump(scenario, f)
                print(f"=> scenario saved to {out_scenario_fpath}")