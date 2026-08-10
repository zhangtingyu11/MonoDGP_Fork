#!/usr/bin/env python3
"""Real KITTI contract and one-epoch benchmark for aligned GIoU loss."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.trainer_helper import CudaBatchPrefetcher  # noqa: E402
from tools.benchmark_foreach_adamw_v2 import (  # noqa: E402
    BoardMemorySampler,
    build_stack,
    environment_receipt,
    load_config as load_seq6_config,
    state_sha256,
)


def load_config(aligned):
    cfg = load_seq6_config(True)
    cfg['model']['use_aligned_giou_loss'] = bool(aligned)
    cfg['trainer']['max_epoch'] = 1
    cfg['trainer']['save_path'] = '/tmp/monodgp_seq7_no_artifacts/'
    return cfg


def model_box_cases(outputs):
    for index, item in enumerate(outputs['inter_outputs']):
        yield f'inter_{index}', item
    yield 'final', outputs
    for index, item in enumerate(outputs.get('aux_outputs', ())):
        yield f'aux_{index}', item


def verify_real_batches(output_path, batch_count):
    cfg = load_config(False)
    train_loader, model, criterion, _, trainer = build_stack(
        cfg, 'seq7-real-contract')
    batch_source = CudaBatchPrefetcher(
        iter(train_loader), torch.device('cuda'),
        copy_stream=trainer.cuda_batch_copy_stream)
    batches_checked = 0
    cases_checked = 0
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
            for _, source in model_box_cases(outputs):
                case = {
                    'pred_logits': source['pred_logits'].detach(),
                    'pred_boxes': source['pred_boxes'].detach().clone(
                        ).requires_grad_(True),
                }
                indices = criterion.matcher(
                    case, targets, group_num=group_num)
                matched_pairs_checked += sum(
                    len(source_indices)
                    for source_indices, _ in indices)

                criterion.use_aligned_giou_loss = False
                ordinary = criterion.loss_boxes(
                    case, targets, indices, num_boxes)
                ordinary_gradient = torch.autograd.grad(
                    ordinary['loss_giou'], case['pred_boxes'],
                    retain_graph=True)[0]
                criterion.use_aligned_giou_loss = True
                aligned = criterion.loss_boxes(
                    case, targets, indices, num_boxes)
                aligned_gradient = torch.autograd.grad(
                    aligned['loss_giou'], case['pred_boxes'])[0]

                if not torch.equal(
                        ordinary['loss_bbox'], aligned['loss_bbox']):
                    raise RuntimeError('L1 box loss changed')
                if not torch.equal(
                        ordinary['loss_giou'], aligned['loss_giou']):
                    raise RuntimeError('GIoU loss changed')
                if not torch.equal(ordinary_gradient, aligned_gradient):
                    raise RuntimeError('GIoU output gradient changed')
                cases_checked += 1
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
        'loss_cases_checked': cases_checked,
        'matched_pairs_checked': matched_pairs_checked,
        'layers_per_batch': 6,
        'l1_loss_exact': True,
        'giou_loss_exact': True,
        'giou_output_gradient_exact': True,
        **environment_receipt(),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def benchmark_arm(arm, output_path):
    aligned = arm == 'aligned'
    cfg = load_config(aligned)
    train_loader, model, _, optimizer, trainer = build_stack(
        cfg, f'seq7-{arm}')
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
        'use_aligned_giou_loss': aligned,
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
        'mode', choices=('verify-real', 'ordinary', 'aligned'))
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
