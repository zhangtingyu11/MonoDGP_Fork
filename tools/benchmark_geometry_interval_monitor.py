"""Real-B16 contract and microbenchmark for feasible-interval monitoring."""

import argparse
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import yaml
from torch.utils.data import DataLoader

import lib.models.monodgp.backbone as backbone_module
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.helpers.model_helper import build_model
from lib.helpers.swanlab_helper import GeometryIntervalAccumulator
from lib.losses.asymmetric_interval_depth_loss import (
    asymmetric_interval_and_uncertainty_loss)
from tools.smoke_geometry_interval_monitor import prepare_targets


def timed_call(callable_):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = callable_()
    end.record()
    end.synchronize()
    return result, start.elapsed_time(end)


def assert_equal(first, second):
    if not torch.equal(first[0], second[0]):
        raise RuntimeError('interval loss changed')
    if not torch.equal(first[1], second[1]):
        raise RuntimeError('uncertainty loss changed')
    if first[2].keys() != second[2].keys():
        raise RuntimeError('receipt keys changed')
    changed = [
        key for key in first[2]
        if not torch.equal(first[2][key], second[2][key])]
    if changed:
        raise RuntimeError(f'receipt values changed: {changed}')


def legacy_accumulator_summary(receipts):
    count_keys = GeometryIntervalAccumulator._COUNT_KEYS
    vector_metrics = GeometryIntervalAccumulator._VECTOR_METRICS
    counts = {key: 0.0 for key in count_keys}
    iou_sum, iou_count = 0.0, 0
    metric_sums = {key: 0.0 for key in vector_metrics}
    metric_counts = {key: 0 for key in vector_metrics}
    widths = {'left_width_virtual': [], 'right_width_virtual': []}
    for receipt in receipts:
        for key in count_keys:
            counts[key] += float(receipt[key].detach().cpu())
        iou = receipt['iou_at_gt'].detach().float().reshape(-1).cpu()
        iou_sum += float(iou.sum())
        iou_count += iou.numel()
        for metric, field in vector_metrics.items():
            values = receipt[field].detach().float().reshape(-1).cpu()
            metric_sums[metric] += float(values.sum())
            metric_counts[metric] += values.numel()
        for field in widths:
            values = receipt[field].detach().float().reshape(-1).cpu()
            if values.numel():
                widths[field].append(values)
    eligible, valid = counts['eligible_car_count'], counts['valid_interval_count']
    result = dict(counts)
    result.update({
        'valid_interval_fraction': valid / eligible if eligible else 0.0,
        'outside_fraction': counts['outside_count'] / valid if valid else 0.0,
        'mean_iou_at_gt': iou_sum / iou_count if iou_count else 0.0,
    })
    result.update({
        key: (metric_sums[key] / metric_counts[key]
              if metric_counts[key] else 0.0)
        for key in vector_metrics})
    left = torch.cat(widths['left_width_virtual'])
    right = torch.cat(widths['right_width_virtual'])
    for name, values in (
            ('left_width_virtual', left),
            ('right_width_virtual', right),
            ('total_width_virtual', left + right)):
        quantiles = torch.quantile(
            values, values.new_tensor((.1, .5, .9))).tolist()
        result[f'{name}_p10'] = float(quantiles[0])
        result[f'{name}_median'] = float(quantiles[1])
        result[f'{name}_p90'] = float(quantiles[2])
    return result


def assert_accumulator_equal(receipts):
    accumulator = GeometryIntervalAccumulator()
    for receipt in receipts:
        accumulator.add(receipt)
    legacy = legacy_accumulator_summary(receipts)
    deferred = accumulator.finalize()
    exact_keys = set(GeometryIntervalAccumulator._COUNT_KEYS)
    exact_keys.update(('valid_interval_fraction', 'outside_fraction'))
    changed = {}
    for key in legacy:
        difference = abs(legacy[key] - deferred[key])
        tolerance = 0.0 if key in exact_keys else 1e-6
        if difference > tolerance:
            changed[key] = {
                'legacy': legacy[key], 'deferred': deferred[key],
                'absolute_difference': difference,
                'tolerance': tolerance,
            }
    if changed:
        raise RuntimeError(f'deferred accumulator changed values: {changed}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--repeats', type=int, default=8)
    args = parser.parse_args()
    with open('configs/monodgp.yaml', encoding='utf-8') as handle:
        cfg = yaml.load(handle, Loader=yaml.Loader)
    dataset_cfg = dict(cfg['dataset'])
    dataset_cfg['batch_size'] = 16
    dataset = KITTI_Dataset('train', dataset_cfg)
    loader = DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=4,
        pin_memory=True, drop_last=False)
    backbone_module.is_main_process = lambda: False
    model, criterion = build_model(cfg['model'])
    checkpoint = torch.load(
        args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    device = torch.device('cuda')
    model.to(device).train()
    criterion.to(device).train()
    inputs, calibs, raw_targets, info = next(iter(loader))
    with torch.no_grad():
        outputs = model(
            inputs.to(device, non_blocking=True),
            calibs.to(device, non_blocking=True), raw_targets,
            info['img_size'].to(device, non_blocking=True), dn_args=0)
        targets = prepare_targets(raw_targets, device)
        final_outputs = {
            key: value for key, value in outputs.items()
            if key not in ('aux_outputs', 'inter_outputs')}
        prepared = (
            criterion.matcher.prepare_targets(targets)
            if getattr(
                criterion.matcher, 'use_batched_same_image_cost', False)
            else None)
        indices = criterion.matcher(
            final_outputs, targets, group_num=criterion.group_num,
            prepared_targets=prepared)
        num_boxes = max(
            sum(len(target['labels']) for target in targets)
            * criterion.group_num, 1)
        kwargs = dict(
            outputs=outputs, targets=targets, indices=indices,
            num_boxes=num_boxes,
            car_class_id=criterion.geometry_interval_monitoring_car_class_id,
            iou_threshold=(
                criterion.geometry_interval_monitoring_iou_threshold),
            decode_mean_sizes=(
                criterion.geometry_interval_monitoring_decode_means))

        def legacy():
            return asymmetric_interval_and_uncertainty_loss(
                **kwargs, fuse_bidirectional_bisection=False,
                reuse_static_iou_geometry=False)

        def fused():
            return asymmetric_interval_and_uncertainty_loss(
                **kwargs, fuse_bidirectional_bisection=True,
                reuse_static_iou_geometry=False)

        def optimized():
            return asymmetric_interval_and_uncertainty_loss(
                **kwargs, fuse_bidirectional_bisection=True,
                reuse_static_iou_geometry=True)

        legacy_receipt = legacy()
        assert_equal(legacy_receipt, fused())
        assert_equal(legacy_receipt, optimized())
        assert_accumulator_equal(
            [legacy_receipt[2], legacy_receipt[2], legacy_receipt[2]])
        for _ in range(2):
            legacy()
            fused()
            optimized()
        legacy_ms, fused_ms, optimized_ms = [], [], []
        for _ in range(args.repeats):
            first, elapsed = timed_call(legacy)
            legacy_ms.append(elapsed)
            second, elapsed = timed_call(fused)
            fused_ms.append(elapsed)
            assert_equal(first, second)
            third, elapsed = timed_call(optimized)
            optimized_ms.append(elapsed)
            assert_equal(first, third)
    legacy_median = statistics.median(legacy_ms)
    fused_median = statistics.median(fused_ms)
    optimized_median = statistics.median(optimized_ms)
    print(f'legacy_ms={legacy_ms}')
    print(f'fused_ms={fused_ms}')
    print(f'optimized_ms={optimized_ms}')
    print(f'legacy_median_ms={legacy_median:.6f}')
    print(f'fused_median_ms={fused_median:.6f}')
    print(f'optimized_median_ms={optimized_median:.6f}')
    print(f'fused_speedup={(legacy_median / fused_median):.6f}x')
    print(f'optimized_speedup={(legacy_median / optimized_median):.6f}x')
    print('GEOMETRY_INTERVAL_B16_CONTRACT_OK')


if __name__ == '__main__':
    main()
