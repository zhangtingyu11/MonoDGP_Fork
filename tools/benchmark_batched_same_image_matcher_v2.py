#!/usr/bin/env python3
"""Real KITTI contract and one-epoch benchmark for same-image matching."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.trainer_helper import CudaBatchPrefetcher  # noqa: E402
from tools.benchmark_aligned_giou_v2 import model_box_cases  # noqa: E402
from tools.benchmark_foreach_adamw_v2 import (  # noqa: E402
    BoardMemorySampler,
    build_stack,
    environment_receipt,
    state_sha256,
)
from tools.benchmark_post_match_cache_v2 import (  # noqa: E402
    load_config as load_seq8_config,
)


def load_config(batched):
    cfg = load_seq8_config(True)
    cfg['model']['use_batched_same_image_matcher_cost'] = bool(batched)
    cfg['trainer']['max_epoch'] = 1
    cfg['trainer']['save_path'] = '/tmp/monodgp_seq9_no_artifacts/'
    return cfg


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        clone = value.detach().clone()
        if clone.is_floating_point() or clone.is_complex():
            clone.requires_grad_(True)
        return clone
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    return value


def tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensor_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensor_leaves(item)


def indices_exact(left, right):
    return (len(left) == len(right)
            and all(torch.equal(left_source, right_source)
                    and torch.equal(left_target, right_target)
                    for (left_source, left_target),
                    (right_source, right_target) in zip(left, right)))


def verify_real_batches(output_path, batch_count):
    cfg = load_config(False)
    train_loader, model, criterion, _, trainer = build_stack(
        cfg, 'seq9-real-contract')
    matcher = criterion.matcher
    batch_source = CudaBatchPrefetcher(
        iter(train_loader), torch.device('cuda'),
        copy_stream=trainer.cuda_batch_copy_stream)
    batches_checked = 0
    layers_checked = 0
    images_checked = 0
    groups_checked = 0
    solver_calls_checked = 0
    loss_values_checked = 0
    output_gradients_checked = 0
    try:
        for inputs, calibs, targets, _ in batch_source:
            img_sizes = targets['img_size']
            targets = trainer.prepare_targets(targets, inputs.shape[0])
            with torch.no_grad():
                outputs = model(
                    inputs, calibs, targets, img_sizes, dn_args=None)

            matcher.use_batched_same_image_cost = False
            ordinary_layers = []
            for _, source in model_box_cases(outputs):
                ordinary_layers.append(matcher(
                    source, targets, group_num=criterion.group_num))

            matcher.use_batched_same_image_cost = True
            prepared = matcher.prepare_targets(targets)
            batched_layers = []
            for _, source in model_box_cases(outputs):
                batched_layers.append(matcher(
                    source, targets, group_num=criterion.group_num,
                    prepared_targets=prepared))

            for ordinary, batched in zip(ordinary_layers, batched_layers):
                if not indices_exact(ordinary, batched):
                    raise RuntimeError('real per-image matcher indices changed')
                layers_checked += 1
                images_checked += len(targets)
                groups_checked += len(targets) * criterion.group_num
                solver_calls_checked += len(targets) * criterion.group_num

            ordinary_outputs = clone_tree(outputs)
            batched_outputs = clone_tree(outputs)
            matcher.use_batched_same_image_cost = False
            ordinary_losses = criterion(ordinary_outputs, targets)
            matcher.use_batched_same_image_cost = True
            batched_losses = criterion(batched_outputs, targets)
            if ordinary_losses.keys() != batched_losses.keys():
                raise RuntimeError('loss keys changed')
            for key in ordinary_losses:
                if not torch.equal(ordinary_losses[key], batched_losses[key]):
                    raise RuntimeError(f'loss changed: {key}')
                loss_values_checked += 1

            ordinary_total = sum(
                value * criterion.weight_dict[key]
                for key, value in ordinary_losses.items()
                if key in criterion.weight_dict)
            batched_total = sum(
                value * criterion.weight_dict[key]
                for key, value in batched_losses.items()
                if key in criterion.weight_dict)
            ordinary_leaves = tuple(tensor_leaves(ordinary_outputs))
            batched_leaves = tuple(tensor_leaves(batched_outputs))
            ordinary_gradients = torch.autograd.grad(
                ordinary_total, ordinary_leaves, allow_unused=True)
            batched_gradients = torch.autograd.grad(
                batched_total, batched_leaves, allow_unused=True)
            for old, new in zip(ordinary_gradients, batched_gradients):
                if old is None or new is None:
                    if old is not None or new is not None:
                        raise RuntimeError('gradient connectivity changed')
                    continue
                if not torch.equal(old, new):
                    raise RuntimeError('combined output gradient changed')
                output_gradients_checked += 1

            batches_checked += 1
            if batches_checked >= batch_count:
                break
    finally:
        matcher.use_batched_same_image_cost = bool(
            cfg['model']['use_batched_same_image_matcher_cost'])
        batch_source.close()
    result = {
        'status': 'passed',
        'real_batches_checked': batches_checked,
        'batch_size': 16,
        'real_images_checked': batches_checked * 16,
        'matcher_layers_checked': layers_checked,
        'per_layer_images_checked': images_checked,
        'query_groups_checked': groups_checked,
        'solver_calls_checked': solver_calls_checked,
        'loss_values_checked': loss_values_checked,
        'output_gradients_checked': output_gradients_checked,
        'all_matcher_indices_exact': True,
        'all_loss_values_exact': True,
        'all_output_gradients_exact': True,
        **environment_receipt(),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def benchmark_arm(arm, output_path):
    batched = arm == 'batched'
    cfg = load_config(batched)
    train_loader, model, _, _, trainer = build_stack(
        cfg, f'seq9-{arm}')
    initial_hash = state_sha256(model)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with BoardMemorySampler() as board:
        start = time.perf_counter()
        summary = trainer.train_one_epoch(0)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
    result = {
        'status': 'completed',
        'arm': arm,
        'use_batched_same_image_matcher_cost': batched,
        'batch_size': 16,
        'workers': train_loader.num_workers,
        'images': len(train_loader.dataset),
        'steps': len(train_loader),
        'seconds': seconds,
        'images_per_second': len(train_loader.dataset) / seconds,
        'mean_loss': summary['mean_loss'],
        'initial_model_sha256': initial_hash,
        'final_model_sha256': state_sha256(model),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'board_fb_peak_mib': board.peak_mib,
        **environment_receipt(),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode', choices=('verify-real', 'ordinary', 'batched'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--batches', type=int, default=12)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.mode == 'verify-real':
        verify_real_batches(args.output, args.batches)
    else:
        benchmark_arm(args.mode, args.output)


if __name__ == '__main__':
    main()
