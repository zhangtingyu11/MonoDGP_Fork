#!/usr/bin/env python3
"""Real KITTI contract and one-epoch benchmark for post-match reuse."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.trainer_helper import CudaBatchPrefetcher  # noqa: E402
from lib.models.monodgp.monodgp import build_post_match_cache  # noqa: E402
from tools.benchmark_aligned_giou_v2 import (  # noqa: E402
    load_config as load_seq7_config,
    model_box_cases,
)
from tools.benchmark_foreach_adamw_v2 import (  # noqa: E402
    BoardMemorySampler,
    build_stack,
    environment_receipt,
    state_sha256,
)


def load_config(cached):
    cfg = load_seq7_config(True)
    cfg['model']['use_post_match_cache'] = bool(cached)
    cfg['trainer']['max_epoch'] = 1
    cfg['trainer']['save_path'] = '/tmp/monodgp_seq8_no_artifacts/'
    return cfg


def affected_losses(criterion, layer_name):
    if layer_name.startswith('inter_'):
        return tuple(criterion.inter_losses)
    return tuple(
        loss for loss in criterion.losses
        if loss not in ('cardinality', 'depth_map', 'region'))


def detached_case(source):
    return {
        key: value.detach().clone().requires_grad_(True)
        for key, value in source.items()
        if isinstance(value, torch.Tensor)
    }


def differentiable_tensors(case):
    return tuple(value for value in case.values() if value.requires_grad)


def verify_real_batches(output_path, batch_count):
    cfg = load_config(False)
    train_loader, model, criterion, _, trainer = build_stack(
        cfg, 'seq8-real-contract')
    batch_source = CudaBatchPrefetcher(
        iter(train_loader), torch.device('cuda'),
        copy_stream=trainer.cuda_batch_copy_stream)
    batches_checked = 0
    layers_checked = 0
    loss_values_checked = 0
    output_gradients_checked = 0
    matched_pairs_checked = 0
    try:
        for inputs, calibs, targets, _ in batch_source:
            img_sizes = targets['img_size']
            targets = trainer.prepare_targets(targets, inputs.shape[0])
            with torch.no_grad():
                outputs = model(
                    inputs, calibs, targets, img_sizes, dn_args=None)
            group_num = criterion.group_num
            num_boxes = max(
                sum(len(target['labels']) for target in targets) * group_num,
                1)
            for layer_name, source in model_box_cases(outputs):
                case = detached_case(source)
                indices = criterion.matcher(
                    case, targets, group_num=group_num)
                matched_pairs_checked += sum(
                    len(source_indices) for source_indices, _ in indices)
                active_losses = affected_losses(criterion, layer_name)
                cache = build_post_match_cache(
                    case, targets, indices, active_losses)
                leaves = differentiable_tensors(case)

                for loss_name in active_losses:
                    ordinary = criterion.get_loss(
                        loss_name, case, targets, indices, num_boxes)
                    cached = criterion.get_loss(
                        loss_name, case, targets, indices, num_boxes,
                        matched_cache=cache)
                    if ordinary.keys() != cached.keys():
                        raise RuntimeError(
                            f'{layer_name}/{loss_name} loss keys changed')
                    for key in ordinary:
                        if not torch.equal(ordinary[key], cached[key]):
                            raise RuntimeError(
                                f'{layer_name}/{loss_name}/{key} changed')
                        loss_values_checked += 1
                        if not ordinary[key].requires_grad:
                            continue
                        ordinary_gradient = torch.autograd.grad(
                            ordinary[key], leaves, retain_graph=True,
                            allow_unused=True)
                        cached_gradient = torch.autograd.grad(
                            cached[key], leaves, retain_graph=True,
                            allow_unused=True)
                        for old, new in zip(
                                ordinary_gradient, cached_gradient):
                            if old is None or new is None:
                                if old is not None or new is not None:
                                    raise RuntimeError(
                                        f'{layer_name}/{loss_name}/{key} '
                                        'gradient connectivity changed')
                                continue
                            if not torch.equal(old, new):
                                raise RuntimeError(
                                    f'{layer_name}/{loss_name}/{key} '
                                    'output gradient changed')
                            output_gradients_checked += 1
                layers_checked += 1
            batches_checked += 1
            if batches_checked >= batch_count:
                break
    finally:
        batch_source.close()
    result = {
        'status': 'passed',
        'real_batches_checked': batches_checked,
        'batch_size': 16,
        'real_images_checked': batches_checked * 16,
        'layers_checked': layers_checked,
        'loss_values_checked': loss_values_checked,
        'output_gradients_checked': output_gradients_checked,
        'matched_pairs_checked': matched_pairs_checked,
        'loss_values_exact': True,
        'output_gradients_exact': True,
        **environment_receipt(),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def benchmark_arm(arm, output_path):
    cached = arm == 'cached'
    cfg = load_config(cached)
    train_loader, model, _, _, trainer = build_stack(
        cfg, f'seq8-{arm}')
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
        'use_post_match_cache': cached,
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
        'mode', choices=('verify-real', 'ordinary', 'cached'))
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
