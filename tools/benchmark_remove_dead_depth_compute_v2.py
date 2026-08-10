#!/usr/bin/env python3
"""Real-B16 contract and speed benchmark for removing dead depth work."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import types

import torch
import torch.nn.functional as F
import torchvision


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_config():
    from tools.benchmark_batched_same_image_matcher_v2 import (
        load_config as load_seq9_config,
    )

    cfg = load_seq9_config(True)
    cfg['trainer']['max_epoch'] = 1
    cfg['trainer']['save_frequency'] = 1000
    cfg['trainer']['save_all'] = False
    cfg['trainer']['save_path'] = '/tmp/monodgp_seq12_no_artifacts/'
    return cfg


def environment_receipt(command):
    import numba
    import numba_cuda

    working_tree_diff = subprocess.run(
        ['git', 'diff'], cwd=ROOT, check=True,
        capture_output=True).stdout
    return {
        'python_executable': sys.executable,
        'python': sys.version,
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'torch_cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'numba': numba.__version__,
        'numba_cuda': numba_cuda.__version__,
        'git_commit': subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip(),
        'working_tree_diff_sha256': hashlib.sha256(
            working_tree_diff).hexdigest(),
        'gpu': torch.cuda.get_device_name(0),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'command': command,
    }


def legacy_depth_forward(self, feature, mask, pos):
    """Exact pre-change DepthPredictor.forward used only by this benchmark."""
    src_16 = self.proj(feature[1])
    src_32 = self.upsample(F.interpolate(
        feature[2], size=src_16.shape[-2:], mode='bilinear'))
    src_8 = self.downsample(feature[0])
    src = (src_8 + src_16 + src_32) / 3

    src = self.depth_head(src)
    depth_logits = self.depth_classifier(src)
    depth_probs = F.softmax(depth_logits, dim=1)
    weighted_depth = (
        depth_probs
        * self.depth_bin_values.reshape(1, -1, 1, 1)
    ).sum(dim=1)

    batch_size, channels, height, width = src.shape
    src = src.flatten(2).permute(2, 0, 1)
    mask = mask.flatten(1)
    pos = pos.flatten(2).permute(2, 0, 1)
    depth_embed = self.depth_encoder(src, mask, pos)
    depth_embed = depth_embed.permute(1, 2, 0).reshape(
        batch_size, channels, height, width)

    # This value was computed but never added to depth_embed.
    self.interpolate_depth_embed(weighted_depth)
    return depth_logits, depth_embed


def select_depth_path(model, legacy):
    module = model.depth_predictor
    if legacy:
        module.forward = types.MethodType(legacy_depth_forward, module)
    else:
        module.forward = types.MethodType(type(module).forward, module)


def clone_state_dict(model):
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


def capture_tree(value, prefix='output'):
    rows = {}
    if torch.is_tensor(value):
        rows[prefix] = value.detach().cpu().clone()
    elif isinstance(value, dict):
        for key, item in value.items():
            rows.update(capture_tree(item, f'{prefix}.{key}'))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            rows.update(capture_tree(item, f'{prefix}[{index}]'))
    return rows


def capture_gradients(model):
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def tensor_metrics(reference, candidate):
    if reference.keys() != candidate.keys():
        missing = sorted(reference.keys() - candidate.keys())
        extra = sorted(candidate.keys() - reference.keys())
        raise RuntimeError(
            f'tensor sets differ; missing={missing[:4]}, extra={extra[:4]}')
    diff_sq = 0.0
    reference_sq = 0.0
    candidate_sq = 0.0
    dot = 0.0
    max_abs = 0.0
    elements = 0
    changed = 0
    for name in reference:
        left = reference[name].double()
        right = candidate[name].double()
        delta = right - left
        diff_sq += float(delta.square().sum())
        reference_sq += float(left.square().sum())
        candidate_sq += float(right.square().sum())
        dot += float((left * right).sum())
        max_abs = max(max_abs, float(delta.abs().max()))
        elements += left.numel()
        changed += int(torch.count_nonzero(delta))
    reference_norm = math.sqrt(reference_sq)
    candidate_norm = math.sqrt(candidate_sq)
    return {
        'relative_l2': math.sqrt(diff_sq) / max(reference_norm, 1e-30),
        'cosine_similarity': (
            dot / max(reference_norm * candidate_norm, 1e-30)),
        'max_absolute_difference': max_abs,
        'changed_elements': changed,
        'elements': elements,
        'changed_fraction': changed / max(elements, 1),
    }


def prepare_batch(trainer, batch):
    inputs, calibs, raw_targets, _ = batch
    targets = trainer.prepare_targets(raw_targets, inputs.shape[0])
    img_sizes = raw_targets['img_size']
    dn_args = None
    if trainer.cfg['use_dn']:
        dn_args = (
            targets, trainer.cfg['scalar'],
            trainer.cfg['label_noise_scale'],
            trainer.cfg['box_noise_scale'],
            trainer.cfg['num_patterns'])
    return inputs, calibs, targets, img_sizes, dn_args


def run_trial(model, criterion, trainer, state, batch, legacy):
    model.load_state_dict(state)
    select_depth_path(model, legacy)
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(20260810)
    torch.cuda.manual_seed_all(20260810)
    inputs, calibs, targets, img_sizes, dn_args = prepare_batch(
        trainer, batch)
    outputs = model(inputs, calibs, targets, img_sizes, dn_args=dn_args)
    losses = criterion(outputs, targets, None)
    total = sum(
        value * criterion.weight_dict[key]
        for key, value in losses.items()
        if key in criterion.weight_dict)
    total.backward()
    torch.cuda.synchronize()
    loss_tensors = {
        **{f'loss.{key}': value.detach().cpu().clone()
           for key, value in losses.items()},
        'loss.__weighted_total__': total.detach().cpu().clone(),
    }
    return {
        'outputs': capture_tree(outputs),
        'losses': loss_tensors,
        'gradients': capture_gradients(model),
    }


def verify_real_batches(output_path, batch_count):
    from lib.helpers.trainer_helper import CudaBatchPrefetcher
    from tools.benchmark_foreach_adamw_v2 import build_stack, state_sha256

    cfg = load_config()
    train_loader, model, criterion, _, trainer = build_stack(
        cfg, 'seq12-real-contract')
    initial_hash = state_sha256(model)
    state = clone_state_dict(model)
    source = CudaBatchPrefetcher(
        iter(train_loader), torch.device('cuda'),
        copy_stream=trainer.cuda_batch_copy_stream)
    comparisons = []
    try:
        for batch_index, batch in enumerate(source):
            legacy_first = run_trial(
                model, criterion, trainer, state, batch, legacy=True)
            legacy_repeat = run_trial(
                model, criterion, trainer, state, batch, legacy=True)
            candidate = run_trial(
                model, criterion, trainer, state, batch, legacy=False)
            comparisons.append({
                'batch_index': batch_index,
                'legacy_repeat': {
                    key: tensor_metrics(legacy_first[key], legacy_repeat[key])
                    for key in legacy_first
                },
                'legacy_vs_candidate': {
                    key: tensor_metrics(legacy_first[key], candidate[key])
                    for key in legacy_first
                },
            })
            if batch_index + 1 >= batch_count:
                break
    finally:
        source.close()
    result = {
        'status': 'completed',
        'mode': 'real_b16_numerical_contract',
        'real_batches': len(comparisons),
        'real_images': len(comparisons) * 16,
        'batch_size': 16,
        'initial_model_sha256': initial_hash,
        'comparisons': comparisons,
        **environment_receipt(
            f'{sys.executable} tools/benchmark_remove_dead_depth_compute_v2.py '
            f'verify --batches {batch_count} --output {output_path}'),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def benchmark_arm(arm, output_path):
    from tools.benchmark_foreach_adamw_v2 import (
        BoardMemorySampler,
        build_stack,
        state_sha256,
    )

    legacy = arm == 'legacy'
    cfg = load_config()
    train_loader, model, _, _, trainer = build_stack(
        cfg, f'seq12-{arm}')
    select_depth_path(model, legacy)
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
        'mode': 'one_epoch_speed',
        'arm': arm,
        'executes_unused_weighted_depth_branch': legacy,
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
        **environment_receipt(
            f'{sys.executable} tools/benchmark_remove_dead_depth_compute_v2.py '
            f'{arm} --output {output_path}'),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def summarize(verification_path, legacy_path, candidate_path, output_path):
    verification = json.loads(
        Path(verification_path).read_text(encoding='utf-8'))
    legacy = json.loads(Path(legacy_path).read_text(encoding='utf-8'))
    candidate = json.loads(Path(candidate_path).read_text(encoding='utf-8'))
    if legacy['initial_model_sha256'] != candidate['initial_model_sha256']:
        raise RuntimeError('speed arms started from different model states')
    result = {
        'status': 'completed',
        'experiment': 'V2-0012 remove unconsumed depth computation',
        'numerical_contract': verification,
        'speed_comparison': {
            'legacy': legacy,
            'candidate': candidate,
            'time_change_percent': (
                candidate['seconds'] / legacy['seconds'] - 1.0) * 100.0,
            'throughput_change_percent': (
                candidate['images_per_second']
                / legacy['images_per_second'] - 1.0) * 100.0,
            'peak_allocated_change_bytes': (
                candidate['peak_allocated_bytes']
                - legacy['peak_allocated_bytes']),
            'peak_reserved_change_bytes': (
                candidate['peak_reserved_bytes']
                - legacy['peak_reserved_bytes']),
            'board_peak_change_mib': (
                candidate['board_fb_peak_mib']
                - legacy['board_fb_peak_mib']),
        },
        'limitations': [
            'Each speed arm ran once, so run-to-run timing variance is not estimated.',
            'The contract uses real B16 batches and compares all model outputs, losses, and gradients.',
            'Legacy self-repeat quantifies ordinary nondeterministic backward noise.',
            'No checkpoint, prediction file, or AP evaluation was produced.',
        ],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode', choices=('verify', 'legacy', 'candidate', 'summarize'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--batches', type=int, default=4)
    parser.add_argument('--verification')
    parser.add_argument('--legacy')
    parser.add_argument('--candidate')
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.mode == 'verify':
        verify_real_batches(args.output, args.batches)
    elif args.mode in ('legacy', 'candidate'):
        benchmark_arm(args.mode, args.output)
    else:
        summarize(
            args.verification, args.legacy, args.candidate, args.output)


if __name__ == '__main__':
    main()
