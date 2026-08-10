"""One-batch contract for the passive feasible-depth interval monitor."""

import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import yaml
from torch.utils.data import DataLoader

import lib.models.monodgp.backbone as backbone_module
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.helpers.model_helper import build_model


OBJECT_KEYS = {
    'labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin',
    'heading_res', 'boxes_3d', 'src_size_3d', 'depth_unit_scale',
    'projective_rotation_y'}
DENSE_KEYS = {
    'depth_map', 'obj_region', 'img_size', 'projective_input_size',
    'projective_image_effective_calib'}


def prepare_targets(targets, device):
    moved = {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in targets.items()}
    mask = moved['mask_2d']
    result = []
    for batch_index in range(mask.shape[0]):
        item = {}
        for key, value in moved.items():
            if key in OBJECT_KEYS:
                item[key] = value[batch_index][mask[batch_index]]
            elif key in DENSE_KEYS:
                item[key] = value[batch_index]
        result.append(item)
    return result


def run_split(split, model, criterion, dataset_cfg, device):
    dataset = KITTI_Dataset(split=split, cfg=dataset_cfg)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(
        item for item in loader
        if bool(item[2]['mask_2d'].any()))
    inputs, calibs, raw_targets, info = batch
    training = split == 'train'
    model.train(training)
    criterion.train(training)
    with torch.no_grad():
        outputs = model(
            inputs.to(device), calibs.to(device), raw_targets,
            info['img_size'].to(device), dn_args=0)
        targets = prepare_targets(raw_targets, device)
        criterion.geometry_interval_monitoring_enabled = False
        baseline = criterion(outputs, targets, mask_dict=None)
        criterion.geometry_interval_monitoring_enabled = True
        candidate = criterion(outputs, targets, mask_dict=None)
    if baseline.keys() != candidate.keys():
        raise RuntimeError('monitor changed loss keys')
    changed = [
        key for key in baseline
        if not torch.equal(baseline[key], candidate[key])]
    if changed:
        raise RuntimeError(f'monitor changed losses: {changed}')
    receipt = criterion.geometry_conditioned_interval_depth_receipts['final']
    required = {
        'matched_count', 'unique_matched_car_count', 'eligible_car_count',
        'valid_interval_count', 'inside_count', 'outside_count',
        'left_width_virtual', 'right_width_virtual',
        'absolute_error_virtual'}
    missing = sorted(required - receipt.keys())
    if missing:
        raise RuntimeError(f'missing receipt fields: {missing}')
    print(
        f'{split}: losses={len(candidate)} '
        f'matched={int(receipt["matched_count"])} '
        f'unique_cars={int(receipt["unique_matched_car_count"])} '
        f'car_predictions={int(receipt["eligible_car_count"])} '
        f'valid={int(receipt["valid_interval_count"])} '
        f'inside={int(receipt["inside_count"])} '
        f'outside={int(receipt["outside_count"])}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    with open('configs/monodgp.yaml', encoding='utf-8') as handle:
        cfg = yaml.load(handle, Loader=yaml.Loader)
    backbone_module.is_main_process = lambda: False
    model, criterion = build_model(cfg['model'])
    if args.checkpoint:
        checkpoint = torch.load(
            args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state'], strict=True)
    device = torch.device('cuda')
    model.to(device)
    criterion.to(device)
    for split in ('train', 'val'):
        run_split(split, model, criterion, cfg['dataset'], device)
    print('GEOMETRY_INTERVAL_MONITOR_SMOKE_OK')


if __name__ == '__main__':
    main()
