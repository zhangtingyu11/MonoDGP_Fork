#!/usr/bin/env python3
"""Measure numerical and speed impact of CUDA float32 matmul TF32."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch
import torchvision


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def load_config():
    from tools.benchmark_batched_same_image_matcher_v2 import (
        load_config as load_seq9_config,
    )

    cfg = load_seq9_config(True)
    cfg['trainer']['max_epoch'] = 1
    cfg['trainer']['save_path'] = '/tmp/monodgp_seq10_no_artifacts/'
    return cfg


def set_matmul_tf32(enabled):
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)


def environment_receipt():
    import numba
    import numba_cuda

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
        'gpu': torch.cuda.get_device_name(0),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
    }


def clone_state_dict(model):
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


def capture_gradients(model):
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def capture_losses(losses, weights):
    result = {
        key: float(value.detach().cpu())
        for key, value in losses.items()
    }
    result['__weighted_total__'] = float(sum(
        value * weights[key]
        for key, value in losses.items()
        if key in weights).detach().cpu())
    return result


def run_gradient_trial(model, criterion, trainer, state, batch, tf32):
    set_matmul_tf32(tf32)
    model.load_state_dict(state)
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(20260810)
    torch.cuda.manual_seed_all(20260810)
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
    outputs = model(inputs, calibs, targets, img_sizes, dn_args=dn_args)
    losses = criterion(outputs, targets, None)
    weights = criterion.weight_dict
    total = sum(
        value * weights[key]
        for key, value in losses.items()
        if key in weights)
    total.backward()
    torch.cuda.synchronize()
    return capture_losses(losses, weights), capture_gradients(model)


def tensor_metrics(reference, candidate):
    if reference.keys() != candidate.keys():
        raise RuntimeError('gradient parameter sets differ')
    diff_sq = 0.0
    reference_sq = 0.0
    candidate_sq = 0.0
    dot = 0.0
    max_abs = 0.0
    elements = 0
    changed = 0
    sign_disagreements = 0
    both_nonzero = 0
    sign_disagreement_reference_sq = 0.0
    sign_disagreement_candidate_sq = 0.0
    for name in reference:
        left = reference[name].double()
        right = candidate[name].double()
        delta = right - left
        diff_sq += float(torch.sum(delta * delta))
        reference_sq += float(torch.sum(left * left))
        candidate_sq += float(torch.sum(right * right))
        dot += float(torch.sum(left * right))
        max_abs = max(max_abs, float(torch.max(torch.abs(delta))))
        elements += left.numel()
        changed += int(torch.count_nonzero(delta))
        active = (left != 0) & (right != 0)
        both_nonzero += int(torch.count_nonzero(active))
        sign_changed = active & (torch.signbit(left) != torch.signbit(right))
        sign_disagreements += int(torch.count_nonzero(sign_changed))
        sign_disagreement_reference_sq += float(torch.sum(
            left[sign_changed] * left[sign_changed]))
        sign_disagreement_candidate_sq += float(torch.sum(
            right[sign_changed] * right[sign_changed]))
    diff_norm = math.sqrt(diff_sq)
    reference_norm = math.sqrt(reference_sq)
    candidate_norm = math.sqrt(candidate_sq)
    return {
        'relative_l2': diff_norm / max(reference_norm, 1e-30),
        'cosine_similarity': (
            dot / max(reference_norm * candidate_norm, 1e-30)),
        'max_absolute_difference': max_abs,
        'changed_fraction': changed / max(elements, 1),
        'nonzero_sign_disagreement_fraction': (
            sign_disagreements / max(both_nonzero, 1)),
        'sign_disagreement_reference_energy_fraction': (
            sign_disagreement_reference_sq / max(reference_sq, 1e-30)),
        'sign_disagreement_candidate_energy_fraction': (
            sign_disagreement_candidate_sq / max(candidate_sq, 1e-30)),
        'elements': elements,
    }


def scalar_metrics(reference, candidate):
    if reference.keys() != candidate.keys():
        raise RuntimeError('loss key sets differ')
    rows = {}
    for key in reference:
        left = reference[key]
        right = candidate[key]
        rows[key] = {
            'reference': left,
            'candidate': right,
            'absolute_difference': abs(right - left),
            'relative_difference': abs(right - left) / max(abs(left), 1e-30),
        }
    return rows


def verify_numerics(output_path, batch_count):
    from lib.helpers.trainer_helper import CudaBatchPrefetcher
    from tools.benchmark_foreach_adamw_v2 import build_stack, state_sha256

    set_matmul_tf32(False)
    cfg = load_config()
    train_loader, model, criterion, _, trainer = build_stack(
        cfg, 'seq10-numerics')
    initial_hash = state_sha256(model)
    initial_state = clone_state_dict(model)
    batch_source = CudaBatchPrefetcher(
        iter(train_loader), torch.device('cuda'),
        copy_stream=trainer.cuda_batch_copy_stream)
    comparisons = []
    try:
        for batch_index, batch in enumerate(batch_source):
            baseline_1 = run_gradient_trial(
                model, criterion, trainer, initial_state, batch, False)
            baseline_2 = run_gradient_trial(
                model, criterion, trainer, initial_state, batch, False)
            tf32_1 = run_gradient_trial(
                model, criterion, trainer, initial_state, batch, True)
            tf32_2 = run_gradient_trial(
                model, criterion, trainer, initial_state, batch, True)
            comparisons.append({
                'batch_index': batch_index,
                'baseline_repeat': {
                    'losses': scalar_metrics(baseline_1[0], baseline_2[0]),
                    'gradients': tensor_metrics(baseline_1[1], baseline_2[1]),
                },
                'tf32_repeat': {
                    'losses': scalar_metrics(tf32_1[0], tf32_2[0]),
                    'gradients': tensor_metrics(tf32_1[1], tf32_2[1]),
                },
                'baseline_vs_tf32': {
                    'losses': scalar_metrics(baseline_1[0], tf32_1[0]),
                    'gradients': tensor_metrics(baseline_1[1], tf32_1[1]),
                },
            })
            if batch_index + 1 >= batch_count:
                break
    finally:
        batch_source.close()
    set_matmul_tf32(False)
    result = {
        'status': 'completed',
        'mode': 'numerical_gradient_comparison',
        'real_batches': len(comparisons),
        'batch_size': 16,
        'real_images': len(comparisons) * 16,
        'trials_per_batch': {
            'baseline_tf32_off': 2,
            'candidate_tf32_on': 2,
        },
        'initial_model_sha256': initial_hash,
        'comparisons': comparisons,
        **environment_receipt(),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def benchmark_arm(arm, output_path):
    from tools.benchmark_foreach_adamw_v2 import (
        BoardMemorySampler,
        build_stack,
        state_sha256,
    )

    tf32 = arm == 'tf32'
    cfg = load_config()
    train_loader, model, _, _, trainer = build_stack(
        cfg, f'seq10-{arm}')
    # set_random_seed applies the adopted production default, so override it
    # here to keep both benchmark arms reproducible after adoption.
    set_matmul_tf32(tf32)
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


def summarize(numerics_path, baseline_path, tf32_path, output_path):
    numerics = json.loads(Path(numerics_path).read_text(encoding='utf-8'))
    baseline = json.loads(Path(baseline_path).read_text(encoding='utf-8'))
    tf32 = json.loads(Path(tf32_path).read_text(encoding='utf-8'))
    if baseline['initial_model_sha256'] != tf32['initial_model_sha256']:
        raise RuntimeError('speed arms did not start from identical model states')
    speed_change = (tf32['seconds'] / baseline['seconds'] - 1.0) * 100.0
    throughput_change = (
        tf32['images_per_second'] / baseline['images_per_second'] - 1.0
    ) * 100.0
    result = {
        'status': 'completed',
        'experiment': 'V2-0010 TF32 matrix multiplication',
        'numerical_comparison': numerics,
        'speed_comparison': {
            'baseline_tf32_off': baseline,
            'candidate_tf32_on': tf32,
            'time_change_percent': speed_change,
            'throughput_change_percent': throughput_change,
            'peak_allocated_change_bytes': (
                tf32['peak_allocated_bytes'] - baseline['peak_allocated_bytes']),
            'peak_reserved_change_bytes': (
                tf32['peak_reserved_bytes'] - baseline['peak_reserved_bytes']),
            'board_peak_change_mib': (
                tf32['board_fb_peak_mib'] - baseline['board_fb_peak_mib']),
        },
        'limitations': [
            'Each speed arm ran once, so run-to-run timing variance is not estimated.',
            'Gradient comparison measures local one-step perturbation, not final AP.',
            'Baseline and TF32 repeat comparisons estimate the ordinary atomicAdd noise floor.',
        ],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode', choices=('numerics', 'baseline', 'tf32', 'summarize'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--batches', type=int, default=4)
    parser.add_argument('--numerics')
    parser.add_argument('--baseline')
    parser.add_argument('--tf32')
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.mode == 'numerics':
        verify_numerics(args.output, args.batches)
    elif args.mode in ('baseline', 'tf32'):
        benchmark_arm(args.mode, args.output)
    else:
        summarize(args.numerics, args.baseline, args.tf32, args.output)


if __name__ == '__main__':
    main()
