"""CAM_FRONT keyframes, DEVIANT-style KITTI labels, official scene splits.

Reference: abhi1kumar/nuscenes-devkit/python-sdk/nuscenes/scripts/export_kitti.py.
Only Car is exported for the Exp63 Car-only baseline. Images remain original
JPEG symlinks; no lidar is used as a model input. Reference label rounding,
zero-point filtering, and sensor-extrinsic-only transform are intentional.
An existing destination is never overwritten. completion.json marks success.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.kitti import KittiDB
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion


def camera_calibration(nusc, sample):
    cam = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    assert cam['is_key_frame'] and lidar['is_key_frame']
    cc = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    lc = nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
    camera_from_lidar = (
        transform_matrix(cc['translation'], Quaternion(cc['rotation']), inverse=True)
        @ transform_matrix(lc['translation'], Quaternion(lc['rotation']))
    )
    kitti_lidar_to_camera = camera_from_lidar @ Quaternion(
        axis=(0, 0, 1), angle=np.pi / 2).transformation_matrix
    projection = np.zeros((3, 4))
    projection[:, :3] = cc['camera_intrinsic']
    return cam, kitti_lidar_to_camera, projection


def export_sample(nusc, sample):
    cam, transform, projection = camera_calibration(nusc, sample)
    _, boxes, _ = nusc.get_sample_data(
        sample['data']['LIDAR_TOP'], box_vis_level=BoxVisibility.NONE)
    lines, counts = [], Counter()
    for box in boxes:
        ann = nusc.get('sample_annotation', box.token)
        if category_to_detection_name(ann['category_name']) != 'car':
            continue
        counts['car_before_filter'] += 1
        if ann['num_lidar_pts'] + ann['num_radar_pts'] <= 0:
            counts['zero_points'] += 1
            continue
        camera_box = KittiDB.box_nuscenes_to_kitti(
            box, Quaternion(matrix=transform[:3, :3]), transform[:3, 3], Quaternion())
        bbox = KittiDB.project_kitti_box_to_image(
            camera_box, projection, (cam['width'], cam['height']))
        if bbox is None:
            counts['outside_image'] += 1
            continue
        camera_box.score = ann['num_lidar_pts'] + ann['num_radar_pts']
        # Exact reference formatting, including alpha=-10 (not provided).
        # Exp63 constructs heading targets from rotation_y and the physical ray.
        lines.append(KittiDB.box_to_string(
            'Car', camera_box, bbox, truncation=0., occlusion=0))
    counts['exported_car'] = len(lines)
    calib = dict(P0=np.zeros((3, 4)), P1=np.zeros((3, 4)), P2=projection,
                 P3=np.zeros((3, 4)), R0_rect=np.eye(3),
                 Tr_velo_to_cam=transform[:3], Tr_imu_to_velo=np.zeros((3, 4)))
    return cam, lines, calib, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataroot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.dataroot), verbose=True)
    split_scenes = create_splits_scenes()
    scene_to_split = {s: split for split in ('train', 'val') for s in split_scenes[split]}
    assert set(split_scenes['train']).isdisjoint(split_scenes['val'])
    for sub in ('image_2', 'label_2', 'calib'):
        (args.output / 'training' / sub).mkdir(parents=True, exist_ok=True)
    (args.output / 'ImageSets').mkdir()
    manifest = dict(command=[sys.executable, *sys.argv], raw_root=str(args.dataroot.resolve()),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    protocol='DEVIANT CAM_FRONT, Car only, exact devkit label formatting',
                    image_storage='original JPEG symlinks', annotation_metadata_sha256=
                    hashlib.sha256((args.dataroot / 'v1.0-trainval/sample_annotation.json').read_bytes()).hexdigest())
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    ids = {'train': [], 'val': []}
    stats = {s: Counter() for s in ids}
    with (args.output / 'sample_mapping.jsonl').open('w') as mapping:
        for index, sample in enumerate(nusc.sample):
            scene = nusc.get('scene', sample['scene_token'])['name']
            split = scene_to_split[scene]
            image_id = f'{index:06d}'
            cam, labels, calib, counts = export_sample(nusc, sample)
            source = (args.dataroot / cam['filename']).resolve(strict=True)
            (args.output / 'training/image_2' / f'{image_id}.jpg').symlink_to(source)
            (args.output / 'training/label_2' / f'{image_id}.txt').write_text(
                ''.join(line + '\n' for line in labels))
            (args.output / 'training/calib' / f'{image_id}.txt').write_text(''.join(
                key + ': ' + ' '.join(f'{v:.12e}' for v in value.ravel()) + '\n'
                for key, value in calib.items()))
            mapping.write(json.dumps(dict(id=image_id, split=split, scene=scene,
                sample_token=sample['token'], camera_token=cam['token'], image=str(source))) + '\n')
            ids[split].append(image_id)
            stats[split].update(counts)
            stats[split]['images'] += 1
            stats[split]['images_over_50_cars'] += len(labels) > 50
            if index % 500 == 0:
                print(json.dumps(dict(processed=index + 1, total=len(nusc.sample))), flush=True)
    assert (len(ids['train']), len(ids['val'])) == (28130, 6019)
    assert set(ids['train']).isdisjoint(ids['val'])
    for split, split_ids in ids.items():
        (args.output / 'ImageSets' / f'{split}.txt').write_text('\n'.join(split_ids) + '\n')
    report = dict(status='complete', stats=stats, scene_counts={s: len(split_scenes[s]) for s in ids})
    (args.output / 'completion.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
