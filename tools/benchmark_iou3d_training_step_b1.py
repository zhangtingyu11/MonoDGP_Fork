"""Real B1 forward/loss/backward A-B-A for optimized 3D-IoU matching."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import yaml


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.model_helper import build_model
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.utils_helper import set_random_seed


class _Logger:
    def info(self, message, *args):
        if args:
            message = message % args
        print(message, flush=True)


def _prepare_training_targets(raw_targets, device, batch_size):
    moved = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in raw_targets.items()}
    mask = moved['mask_2d']
    object_fields = {
        'labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin',
        'heading_res', 'boxes_3d', 'src_size_3d', 'depth_unit_scale',
        'projective_rotation_y'}
    image_fields = {
        'depth_map', 'obj_region', 'img_size', 'projective_input_size',
        'projective_image_effective_calib'}
    targets = []
    for batch_index in range(batch_size):
        item = {}
        for key, value in moved.items():
            if key in object_fields:
                item[key] = value[batch_index][mask[batch_index]]
            elif key in image_fields:
                item[key] = value[batch_index]
        targets.append(item)
    return targets


def _run_arm(model, criterion, batch, weight, repeats, warmup, device,
             cpu_rng=None, cuda_rng=None):
    inputs, calibs, raw_targets, info = batch
    inputs = inputs.to(device)
    calibs = calibs.to(device)
    img_sizes = info['img_size'].to(device)
    targets = _prepare_training_targets(
        raw_targets, device, inputs.shape[0])
    criterion.matcher.cost_iou3d = float(weight)
    if cpu_rng is None:
        cpu_rng = torch.random.get_rng_state()
    if cuda_rng is None:
        cuda_rng = torch.cuda.get_rng_state(device)

    def step(measure):
        model.zero_grad(set_to_none=True)
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        if measure:
            torch.cuda.reset_peak_memory_stats(device)
            base_memory = torch.cuda.memory_allocated(device)
            start = time.perf_counter()
        outputs = model(inputs, calibs, targets, img_sizes, dn_args=0)
        loss_dict = criterion(outputs, targets, mask_dict=None)
        loss = sum(
            value * criterion.weight_dict[key]
            for key, value in loss_dict.items()
            if key in criterion.weight_dict)
        loss.backward()
        torch.cuda.synchronize(device)
        if not measure:
            return None
        milliseconds = (time.perf_counter() - start) * 1000
        peak = torch.cuda.max_memory_allocated(device)
        finite_gradients = all(
            torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
            if parameter.grad is not None)
        return {
            'milliseconds': milliseconds,
            'incremental_peak_allocated_mib': (peak - base_memory) / 2**20,
            'loss': loss.detach().item(),
            'finite_gradients': finite_gradients,
            'matcher_receipt': dict(criterion.matcher.last_iou3d_receipt),
        }

    for _ in range(warmup):
        step(False)
    samples = [step(True) for _ in range(repeats)]
    times = torch.tensor(
        [sample['milliseconds'] for sample in samples], dtype=torch.float64)
    memories = torch.tensor(
        [sample['incremental_peak_allocated_mib'] for sample in samples],
        dtype=torch.float64)
    return {
        'milliseconds_mean': times.mean().item(),
        'milliseconds_median': times.median().item(),
        'milliseconds_min': times.min().item(),
        'milliseconds_max': times.max().item(),
        'incremental_peak_allocated_mib_max': memories.max().item(),
        'losses': [sample['loss'] for sample in samples],
        'all_gradients_finite': all(
            sample['finite_gradients'] for sample in samples),
        'last_matcher_receipt': samples[-1]['matcher_receipt'],
    }


def _merge_runs(runs):
    times = torch.tensor(
        [run['milliseconds_median'] for run in runs], dtype=torch.float64)
    memories = torch.tensor([
        run['incremental_peak_allocated_mib_max'] for run in runs
    ], dtype=torch.float64)
    return {
        'milliseconds_mean': times.mean().item(),
        'milliseconds_median': times.median().item(),
        'milliseconds_min': times.min().item(),
        'milliseconds_max': times.max().item(),
        'incremental_peak_allocated_mib_max': memories.max().item(),
        'losses': [run['losses'][0] for run in runs],
        'all_gradients_finite': all(
            run['all_gradients_finite'] for run in runs),
        'last_matcher_receipt': runs[-1]['last_matcher_receipt'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as handle:
        cfg = yaml.load(handle, Loader=yaml.Loader)
    cfg['dataset']['batch_size'] = 1
    set_random_seed(cfg.get('random_seed', 444))
    train_loader, _ = build_dataloader(cfg['dataset'], workers=0)
    batch = next(iter(train_loader))
    model, criterion = build_model(cfg['model'])
    device = torch.device('cuda')
    model.to(device).train()
    criterion.to(device).train()
    epoch, best_result, best_epoch = load_checkpoint(
        model, None, args.checkpoint, device, logger=_Logger())

    shared_cpu_rng = torch.random.get_rng_state()
    shared_cuda_rng = torch.cuda.get_rng_state(device)
    # Warm both paths before paired measurement.
    _run_arm(model, criterion, batch, 0, 1, args.warmup, device,
             shared_cpu_rng, shared_cuda_rng)
    _run_arm(model, criterion, batch, 5, 1, args.warmup, device,
             shared_cpu_rng, shared_cuda_rng)
    runs = {0: [], 5: []}
    paired_differences = []
    for repeat_index in range(args.repeats):
        order = (0, 5) if repeat_index % 2 == 0 else (5, 0)
        pair = {}
        for weight in order:
            pair[weight] = _run_arm(
                model, criterion, batch, weight, 1, 0, device,
                shared_cpu_rng, shared_cuda_rng)
            runs[weight].append(pair[weight])
        paired_differences.append(
            pair[5]['milliseconds_median']
            - pair[0]['milliseconds_median'])
    baseline = _merge_runs(runs[0])
    candidate = _merge_runs(runs[5])
    paired = torch.tensor(paired_differences, dtype=torch.float64)
    result = {
        'checkpoint_epoch': int(epoch),
        'checkpoint_best_result': float(best_result),
        'checkpoint_best_epoch': int(best_epoch),
        'image_id': int(batch[3]['img_id'][0]),
        'target_count': int(batch[2]['mask_2d'][0].sum()),
        'baseline': baseline,
        'candidate_iou3d_weight_5': candidate,
        'paired_extra_milliseconds': paired.tolist(),
        'candidate_extra_milliseconds_median': paired.median().item(),
        'candidate_slowdown_fraction': (
            paired.median().item() / baseline['milliseconds_median']),
        'candidate_extra_peak_mib': (
            candidate['incremental_peak_allocated_mib_max']
            - baseline['incremental_peak_allocated_mib_max']),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
