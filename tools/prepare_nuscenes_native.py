"""Index original nuScenes annotations without KITTI conversion or rounding.

JSONL is a lossless metadata cache, not a different annotation convention.
Original global centres, quaternions, sizes and annotation tokens are retained.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from lib.datasets.nuscenes.front_geometry import projected_box_intersection, transform_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataroot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.dataroot), verbose=True)
    splits = create_splits_scenes()
    scene_split = {name: split for split in ('train', 'val') for name in splits[split]}
    args.output.mkdir(parents=True)
    handles = {split: (args.output / f'{split}.jsonl').open('w') for split in ('train', 'val')}
    stats = {split: Counter() for split in handles}
    for index, sample in enumerate(nusc.sample):
        scene = nusc.get('scene', sample['scene_token'])['name']
        split = scene_split[scene]
        camera = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        sensor = nusc.get('calibrated_sensor', camera['calibrated_sensor_token'])
        pose = nusc.get('ego_pose', camera['ego_pose_token'])
        lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_pose = nusc.get('ego_pose', lidar['ego_pose_token'])
        # Official distance evaluation uses the sample's LIDAR_TOP ego position.
        ego_position = np.asarray(lidar_pose['translation'], dtype=np.float64)
        world_from_camera = (transform_matrix(pose['translation'], Quaternion(pose['rotation']))
                             @ transform_matrix(sensor['translation'], Quaternion(sensor['rotation'])))
        camera_from_world = np.linalg.inv(world_from_camera)
        intrinsic = np.asarray(sensor['camera_intrinsic'], dtype=np.float64)
        image_size = (camera['width'], camera['height'])
        annotations = []
        for token in sample['anns']:
            ann = nusc.get('sample_annotation', token)
            if ann['category_name'] != 'vehicle.car':
                continue
            stats[split]['all_car'] += 1
            if np.linalg.norm(np.asarray(ann['translation'])[:2] - ego_position[:2]) >= 50:
                stats[split]['outside_50m'] += 1
                continue
            if ann['num_lidar_pts'] + ann['num_radar_pts'] == 0:
                stats[split]['zero_points'] += 1
                continue
            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
            corners = transform_points(box.corners().T, camera_from_world)
            bbox = projected_box_intersection(corners, intrinsic, image_size)
            if bbox is None:
                stats[split]['outside_fov'] += 1
                continue
            centre = transform_points(np.array([ann['translation']]), camera_from_world)[0]
            rotation = camera_from_world[:3, :3] @ box.rotation_matrix
            heading = rotation[:, 0]
            yaw = float(-np.arctan2(heading[2], heading[0]))
            centre_projection = intrinsic @ centre
            uv = centre_projection[:2] / centre_projection[2] if centre[2] != 0 else np.array([np.inf, np.inf])
            inside = bool(centre[2] > 0 and 0 <= uv[0] < image_size[0] and 0 <= uv[1] < image_size[1])
            stats[split]['retained_gt'] += 1
            stats[split]['centre_outside_image'] += not inside
            stats[split]['centre_depth_below_2m'] += centre[2] < 2
            stats[split]['centre_depth_nonpositive'] += centre[2] <= 0
            stats[split]['bbox_height_below_25px'] += bbox[3] - bbox[1] < 25
            yaw_rotation = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
            angle = float(np.arccos(np.clip((np.trace((yaw_rotation @ np.array([[1,0,0],[0,0,-1],[0,1,0]])).T @ rotation)-1)/2, -1, 1)))
            stats[split]['camera_rotation_residual_over_1deg'] += angle > np.pi / 180
            annotations.append(dict(token=token, translation=ann['translation'], size=ann['size'],
                rotation=ann['rotation'], num_pts=ann['num_lidar_pts']+ann['num_radar_pts'],
                camera_center=centre.tolist(), camera_yaw=yaw, bbox=bbox.tolist(),
                centre_inside_image=inside, camera_rotation_residual_rad=angle))
        record = dict(id=index, sample_token=sample['token'], scene=scene,
            image=str((args.dataroot / camera['filename']).resolve(strict=True)),
            image_size=image_size, intrinsic=intrinsic.tolist(),
            camera_from_world=camera_from_world.tolist(), world_from_camera=world_from_camera.tolist(),
            ego_translation=ego_position.tolist(), camera_timestamp=camera['timestamp'],
            annotation_timestamp=sample['timestamp'], annotations=annotations)
        handles[split].write(json.dumps(record, allow_nan=False) + '\n')
        stats[split]['images'] += 1
        stats[split]['images_over_50_gt'] += len(annotations)>50
        if index % 1000 == 0:
            print(json.dumps(dict(processed=index+1, total=len(nusc.sample))), flush=True)
    for handle in handles.values():
        handle.close()
    assert (stats['train']['images'], stats['val']['images']) == (28130, 6019)
    manifest = dict(status='complete', protocol='nuscenes_front_car_v1',
        stats={split:{key:int(value) for key,value in counts.items()} for split,counts in stats.items()},
        input_root=str(args.dataroot), command=[sys.executable,*sys.argv],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        raw_metadata_sha256={name:hashlib.sha256((args.dataroot/'v1.0-trainval'/name).read_bytes()).hexdigest()
            for name in ['sample.json','sample_annotation.json','ego_pose.json','calibrated_sensor.json']},
        notes=['GT global poses retained exactly; camera-time ego pose used',
               'No moving-object temporal interpolation',
               'FOV: positive-area projected silhouette intersection, 1mm near clipping plane',
               'Car, ego XY distance <50m, nonzero lidar+radar points; no height/occlusion/difficulty filter'])
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)


if __name__ == '__main__':
    main()
