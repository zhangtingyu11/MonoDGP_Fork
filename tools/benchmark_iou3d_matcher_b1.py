"""Correctness and real-B1 timing for optimized 3D-IoU matching."""
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
from lib.models.monodgp.iou3d_match_cost import pairwise_iou3d_match_cost
from lib.models.monodgp.matcher import HungarianMatcher
from tools.audit_3d_iou_match_weight import _pairwise_iou3d, _prepare_targets


class _Logger:
    def info(self, message, *args):
        if args:
            message = message % args
        print(message, flush=True)


def _repeat_groups(outputs, groups=11):
    fields = ('pred_logits', 'pred_boxes', 'pred_depth',
              'pred_3d_dim', 'pred_angle')
    return {
        key: outputs[key].repeat(1, groups, *([1] * (outputs[key].ndim - 2)))
        for key in fields
    }


def _timed(matcher, layers, targets, repeats, warmup, device):
    prepared = matcher.prepare_targets(targets)

    def run():
        return [matcher(
            layer, targets, group_num=11,
            prepared_targets=prepared) for layer in layers]

    for _ in range(warmup):
        run()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_memory = torch.cuda.memory_allocated(device)
    samples = []
    last = None
    receipts = []
    for _ in range(repeats):
        start = time.perf_counter()
        last = run()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
        receipts.append(dict(matcher.last_iou3d_receipt))
    peak = torch.cuda.max_memory_allocated(device)
    values = torch.tensor(samples, dtype=torch.float64)
    return {
        'milliseconds_mean': values.mean().item(),
        'milliseconds_median': values.median().item(),
        'milliseconds_p10': torch.quantile(values, 0.10).item(),
        'milliseconds_p90': torch.quantile(values, 0.90).item(),
        'milliseconds_min': values.min().item(),
        'milliseconds_max': values.max().item(),
        'incremental_peak_allocated_mib': (peak - baseline_memory) / 2**20,
        'indices': [
            [(source.tolist(), target.tolist()) for source, target in indices]
            for indices in last],
        'last_receipt': receipts[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=40)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--minimum-targets', type=int, default=5)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as handle:
        cfg = yaml.load(handle, Loader=yaml.Loader)
    cfg['dataset']['batch_size'] = 1
    set_random_seed(cfg.get('random_seed', 444))
    _, loader = build_dataloader(cfg['dataset'], workers=0)
    model, _ = build_model(cfg['model'])
    device = torch.device('cuda')
    model.to(device).eval()
    epoch, best_result, best_epoch = load_checkpoint(
        model, None, args.checkpoint, device, logger=_Logger())

    chosen = None
    for batch in loader:
        raw_targets = batch[2]
        target_count = int(raw_targets['mask_2d'][0].sum())
        if chosen is None or target_count > chosen[4]:
            chosen = (*batch, target_count)
        if target_count >= args.minimum_targets:
            chosen = (*batch, target_count)
            break
    inputs, calibs, raw_targets, info, target_count = chosen
    inputs = inputs.to(device)
    calibs = calibs.to(device)
    img_sizes = info['img_size'].to(device)
    targets = _prepare_targets(raw_targets, device, 1)
    with torch.inference_mode():
        outputs = model(inputs, calibs, raw_targets, img_sizes, dn_args=0)
    layers = [_repeat_groups(outputs)] + [
        _repeat_groups(layer) for layer in outputs['aux_outputs']]

    means = torch.as_tensor(loader.dataset.cls_mean_size, device=device)
    sparse_iou, sparse_receipt = pairwise_iou3d_match_cost(
        {key: value[0] for key, value in layers[0].items()},
        targets[0], means)
    dense_iou = _pairwise_iou3d(
        {key: value[0] for key, value in layers[0].items()},
        targets[0], loader.dataset.cls_mean_size)
    max_iou_difference = (sparse_iou - dense_iou).abs().max().item()

    kwargs = dict(
        cost_class=cfg['model']['set_cost_class'],
        cost_bbox=cfg['model']['set_cost_bbox'],
        cost_3dcenter=cfg['model']['set_cost_3dcenter'],
        cost_giou=cfg['model']['set_cost_giou'],
        use_batched_same_image_cost=True,
        iou3d_decode_mean_sizes=loader.dataset.cls_mean_size)
    baseline = HungarianMatcher(**kwargs, cost_iou3d=0).to(device)
    candidate = HungarianMatcher(**kwargs, cost_iou3d=5).to(device)
    baseline_result = _timed(
        baseline, layers, targets, args.repeats, args.warmup, device)
    candidate_result = _timed(
        candidate, layers, targets, args.repeats, args.warmup, device)
    # Time the baseline once more to expose drift from concurrent GPU work.
    baseline_repeat = _timed(
        baseline, layers, targets, args.repeats, args.warmup, device)

    result = {
        'checkpoint_epoch': int(epoch),
        'checkpoint_best_result': float(best_result),
        'checkpoint_best_epoch': int(best_epoch),
        'image_id': int(info['img_id'][0]),
        'target_count': target_count,
        'query_count': int(layers[0]['pred_boxes'].shape[1]),
        'three_3d_decoder_layers': len(layers),
        'sparse_exact_pair_receipt': sparse_receipt,
        'dense_sparse_max_absolute_iou_difference': max_iou_difference,
        'baseline_first': baseline_result,
        'candidate_iou3d_weight_5': candidate_result,
        'baseline_repeat': baseline_repeat,
    }
    result['candidate_minus_baseline_median_ms'] = (
        candidate_result['milliseconds_median']
        - 0.5 * (baseline_result['milliseconds_median']
                 + baseline_repeat['milliseconds_median']))
    result['candidate_extra_peak_mib'] = (
        candidate_result['incremental_peak_allocated_mib']
        - max(baseline_result['incremental_peak_allocated_mib'],
              baseline_repeat['incremental_peak_allocated_mib']))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
