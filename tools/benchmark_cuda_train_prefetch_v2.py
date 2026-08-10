#!/usr/bin/env python3
"""Real KITTI correctness and two-epoch benchmark for training prefetch."""

import argparse
import collections
import itertools
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.dataloader_helper import build_dataloader  # noqa: E402
from lib.helpers.model_helper import build_model  # noqa: E402
from lib.helpers.optimizer_helper import build_optimizer  # noqa: E402
from lib.helpers.scheduler_helper import build_lr_scheduler  # noqa: E402
from lib.helpers.tester_helper import Tester  # noqa: E402
from lib.helpers.trainer_helper import (  # noqa: E402
    CudaBatchPrefetcher,
    Trainer,
    _tensor_leaves,
)
from lib.helpers.utils_helper import set_random_seed  # noqa: E402


def load_config():
    with (ROOT / 'configs/monodgp.yaml').open(encoding='utf-8') as handle:
        cfg = yaml.load(handle, Loader=yaml.Loader)
    cfg['dataset']['batch_size'] = 16
    cfg['dataset']['test_pin_memory'] = False
    cfg['tester']['export_predictions'] = False
    cfg['trainer']['max_epoch'] = 2
    cfg['trainer']['save_frequency'] = 1000
    cfg['trainer']['save_all'] = False
    cfg['trainer']['save_path'] = '/tmp/monodgp_seq4_no_artifacts/'
    return cfg


def environment_receipt():
    return {
        'python': sys.version,
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'gpu': torch.cuda.get_device_name(0),
    }


def tensor_tree_items(value, prefix=''):
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from tensor_tree_items(value[key], f'{prefix}/{key}')
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensor_tree_items(item, f'{prefix}/{index}')


def state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(str(tuple(value.shape)).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class BoardMemorySampler:
    def __init__(self):
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None

    def _sample_once(self):
        completed = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used',
             '--format=csv,noheader,nounits', '--id=0'],
            check=True, capture_output=True, text=True)
        return int(completed.stdout.strip().splitlines()[0])

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self.samples.append(self._sample_once())
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self.stop_event.wait(0.25)

    def __enter__(self):
        self.samples.append(self._sample_once())
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        self.thread.join()
        self.samples.append(self._sample_once())

    @property
    def peak_mib(self):
        return max(self.samples) if self.samples else None


class TappedIterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.pending = collections.deque()

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.iterator)
        self.pending.append(batch)
        return batch


class OneBatchLoader:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        return itertools.islice(iter(self.loader), 1)

    def __len__(self):
        return 1


def verify_real_transfer(output_path):
    cfg = load_config()
    seed = cfg.get('random_seed', 444)

    # First record the exact image order produced by the ordinary training
    # loader. Resetting the same seed before constructing the prefetched loader
    # makes this an independent skip/reorder check, not merely a copy check.
    cfg['dataset']['train_pin_memory'] = False
    set_random_seed(seed)
    ordinary_loader, _ = build_dataloader(cfg['dataset'])
    ordinary_order_digest = hashlib.sha256()
    ordinary_batches = 0
    ordinary_images = 0
    for _, _, _, info in ordinary_loader:
        image_ids = torch.as_tensor(info['img_id'])
        ordinary_order_digest.update(image_ids.numpy().tobytes())
        ordinary_images += int(image_ids.numel())
        ordinary_batches += 1

    cfg['dataset']['train_pin_memory'] = True
    set_random_seed(seed)
    train_loader, _ = build_dataloader(cfg['dataset'])
    device = torch.device('cuda')
    copy_stream = torch.cuda.Stream(device=device)
    stream_id = copy_stream.cuda_stream
    tapped = TappedIterator(iter(train_loader))
    prefetcher = CudaBatchPrefetcher(
        tapped, device, copy_stream=copy_stream)

    batches = 0
    images = 0
    tensor_elements = 0
    mismatch_elements = 0
    unpinned_tensors = 0
    prefetch_order_digest = hashlib.sha256()
    try:
        for gpu_batch in prefetcher:
            host_batch = tapped.pending.popleft()
            host_items = list(tensor_tree_items(host_batch[:3]))
            gpu_items = list(tensor_tree_items(gpu_batch[:3]))
            if [item[0] for item in host_items] != [item[0] for item in gpu_items]:
                raise RuntimeError('CPU and GPU batch structures differ')
            for (host_path, host), (_, gpu) in zip(host_items, gpu_items):
                if not host.is_pinned():
                    unpinned_tensors += 1
                restored = gpu.detach().cpu()
                tensor_elements += host.numel()
                if host.dtype.is_floating_point:
                    equal = torch.eq(host, restored)
                    equal |= torch.isnan(host) & torch.isnan(restored)
                    mismatch_elements += int((~equal).sum().item())
                else:
                    mismatch_elements += int((host != restored).sum().item())
                if host.shape != restored.shape or host.dtype != restored.dtype:
                    raise RuntimeError(f'tensor metadata differs at {host_path}')
            image_ids = torch.as_tensor(host_batch[3]['img_id'])
            prefetch_order_digest.update(image_ids.numpy().tobytes())
            images += int(image_ids.numel())
            batches += 1
    finally:
        prefetcher.close()

    result = {
        'status': 'passed' if (
            ordinary_batches == batches == len(train_loader)
            and ordinary_images == images == len(train_loader.dataset)
            and mismatch_elements == 0 and unpinned_tensors == 0
            and ordinary_order_digest.digest()
            == prefetch_order_digest.digest()
            and copy_stream.cuda_stream == stream_id) else 'failed',
        'dataset_images': images,
        'batches': batches,
        'batch_size': cfg['dataset']['batch_size'],
        'tensor_elements_compared': tensor_elements,
        'mismatched_elements': mismatch_elements,
        'unpinned_source_tensors': unpinned_tensors,
        'ordinary_batches': ordinary_batches,
        'ordinary_images': ordinary_images,
        'ordinary_batch_order_sha256': ordinary_order_digest.hexdigest(),
        'prefetch_batch_order_sha256': prefetch_order_digest.hexdigest(),
        'batch_order_exactly_equal': (
            ordinary_order_digest.digest() == prefetch_order_digest.digest()),
        'copy_stream_reused': copy_stream.cuda_stream == stream_id,
        **environment_receipt(),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if result['status'] != 'passed':
        raise RuntimeError(f'real transfer verification failed: {result}')


def benchmark_arm(arm, output_path):
    candidate = arm == 'candidate'
    cfg = load_config()
    cfg['dataset']['train_pin_memory'] = candidate
    cfg['trainer']['use_cuda_batch_prefetch'] = candidate
    seed = cfg.get('random_seed', 444)
    set_random_seed(seed)

    train_loader, test_loader = build_dataloader(cfg['dataset'])
    model, loss = build_model(cfg['model'])
    device = torch.device('cuda')
    model = model.to(device)
    optimizer = build_optimizer(cfg['optimizer'], model)
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(
        cfg['lr_scheduler'], optimizer, last_epoch=-1)
    logger = logging.getLogger(f'seq4-{arm}')
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    trainer = Trainer(
        cfg=cfg['trainer'], model=model, optimizer=optimizer,
        train_loader=train_loader, test_loader=test_loader,
        lr_scheduler=lr_scheduler,
        warmup_lr_scheduler=warmup_lr_scheduler,
        logger=logger, loss=loss, model_name=f'seq4_{arm}')
    initial_hash = state_sha256(model)
    stream_id = (trainer.cuda_batch_copy_stream.cuda_stream
                 if candidate else None)

    epochs = []
    for epoch in range(2):
        np.random.seed(np.random.get_state()[1][0] + epoch)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        with BoardMemorySampler() as board:
            start = time.perf_counter()
            summary = trainer.train_one_epoch(epoch)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
        if warmup_lr_scheduler is not None and epoch < 5:
            warmup_lr_scheduler.step()
        else:
            lr_scheduler.step()
        epochs.append({
            'epoch': epoch + 1,
            'seconds': seconds,
            'images': len(train_loader.dataset),
            'steps': len(train_loader),
            'images_per_second': len(train_loader.dataset) / seconds,
            'mean_loss': summary['mean_loss'],
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'end_allocated_bytes': torch.cuda.memory_allocated(device),
            'end_reserved_bytes': torch.cuda.memory_reserved(device),
            'board_fb_peak_mib': board.peak_mib,
            'copy_stream_id': (
                trainer.cuda_batch_copy_stream.cuda_stream
                if candidate else None),
        })

    final_hash = state_sha256(model)
    tester_cfg = dict(cfg['tester'])
    tester_cfg['export_predictions'] = False
    tester = Tester(
        cfg=tester_cfg, model=model, dataloader=test_loader, logger=logger,
        train_cfg=cfg['trainer'], model_name=f'seq4_{arm}')
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    with BoardMemorySampler() as board:
        validation_start = time.perf_counter()
        predictions = tester.inference()
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - validation_start
        ap_start = time.perf_counter()
        moderate_ap = tester.evaluate(predictions)
        torch.cuda.synchronize()
        ap_seconds = time.perf_counter() - ap_start

    result = {
        'status': 'completed',
        'arm': arm,
        'seed': seed,
        'batch_size': cfg['dataset']['batch_size'],
        'workers': train_loader.num_workers,
        'train_pin_memory': train_loader.pin_memory,
        'use_cuda_batch_prefetch': candidate,
        'initial_model_sha256': initial_hash,
        'final_model_sha256': final_hash,
        'persistent_copy_stream_id': stream_id,
        'epochs': epochs,
        'validation': {
            'inference_seconds': inference_seconds,
            'ap_seconds': ap_seconds,
            'complete_seconds': inference_seconds + ap_seconds,
            'moderate_3d_ap_r40': float(moderate_ap),
            'prediction_images_in_memory': len(predictions),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'board_fb_peak_mib': board.peak_mib,
        },
        **environment_receipt(),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def smoke_candidate(output_path):
    cfg = load_config()
    cfg['dataset']['train_pin_memory'] = True
    cfg['trainer']['use_cuda_batch_prefetch'] = True
    set_random_seed(cfg.get('random_seed', 444))
    train_loader, test_loader = build_dataloader(cfg['dataset'])
    model, loss = build_model(cfg['model'])
    model = model.cuda()
    optimizer = build_optimizer(cfg['optimizer'], model)
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(
        cfg['lr_scheduler'], optimizer, last_epoch=-1)
    logger = logging.getLogger('seq4-smoke')
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    trainer = Trainer(
        cfg=cfg['trainer'], model=model, optimizer=optimizer,
        train_loader=OneBatchLoader(train_loader), test_loader=test_loader,
        lr_scheduler=lr_scheduler,
        warmup_lr_scheduler=warmup_lr_scheduler, logger=logger,
        loss=loss, model_name='seq4_smoke')
    torch.cuda.reset_peak_memory_stats()
    summary = trainer.train_one_epoch(0)
    torch.cuda.synchronize()
    result = {
        'status': 'passed' if (
            summary['batch_count'] == 1
            and np.isfinite(summary['mean_loss'])) else 'failed',
        'batch_count': summary['batch_count'],
        'mean_loss': summary['mean_loss'],
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'copy_stream_id': trainer.cuda_batch_copy_stream.cuda_stream,
        **environment_receipt(),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if result['status'] != 'passed':
        raise RuntimeError(f'candidate smoke failed: {result}')


def _blocking_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {
            key: _blocking_to_device(item, device)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_blocking_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_blocking_to_device(item, device) for item in value]
    return value


def _gradient_difference(reference, candidate):
    differing = 0
    elements = 0
    max_abs = 0.0
    difference_sq = 0.0
    reference_sq = 0.0
    for name in reference:
        ref = reference[name].double()
        other = candidate[name].double()
        diff = ref - other
        differing += int((diff != 0).sum().item())
        elements += ref.numel()
        max_abs = max(max_abs, float(diff.abs().max().item()))
        difference_sq += float(diff.square().sum().item())
        reference_sq += float(ref.square().sum().item())
    return {
        'elements': elements,
        'differing_elements': differing,
        'max_abs_difference': max_abs,
        'difference_l2': difference_sq ** 0.5,
        'reference_l2': reference_sq ** 0.5,
        'relative_l2': ((difference_sq / reference_sq) ** 0.5
                        if reference_sq else 0.0),
    }


def verify_gradient_noise(output_path):
    cfg = load_config()
    cfg['dataset']['train_pin_memory'] = True
    cfg['trainer']['use_cuda_batch_prefetch'] = True
    set_random_seed(cfg.get('random_seed', 444))
    train_loader, test_loader = build_dataloader(cfg['dataset'])
    host_batch = next(iter(train_loader))
    model, loss_module = build_model(cfg['model'])
    model = model.cuda().train()
    optimizer = build_optimizer(cfg['optimizer'], model)
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(
        cfg['lr_scheduler'], optimizer, last_epoch=-1)
    logger = logging.getLogger('seq4-gradient-noise')
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    trainer = Trainer(
        cfg=cfg['trainer'], model=model, optimizer=optimizer,
        train_loader=train_loader, test_loader=test_loader,
        lr_scheduler=lr_scheduler,
        warmup_lr_scheduler=warmup_lr_scheduler, logger=logger,
        loss=loss_module, model_name='seq4_gradient_noise')
    initial_state_hash = state_sha256(model)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()

    def run_pass(mode):
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        model.zero_grad(set_to_none=True)
        if mode == 'blocking':
            moved = _blocking_to_device(host_batch[:3], torch.device('cuda'))
            batch = (*moved, host_batch[3])
        else:
            prefetcher = CudaBatchPrefetcher(
                iter([host_batch]), torch.device('cuda'),
                copy_stream=trainer.cuda_batch_copy_stream)
            batch = next(prefetcher)
            prefetcher.close()
        inputs, calibs, targets, _ = batch
        img_sizes = targets['img_size']
        prepared = trainer.prepare_targets(targets, inputs.shape[0])
        outputs = model(
            inputs, calibs, prepared, img_sizes, dn_args=None)
        losses = loss_module(outputs, prepared, None)
        total = sum(
            losses[key] * loss_module.weight_dict[key]
            for key in losses if key in loss_module.weight_dict)
        total.backward()
        torch.cuda.synchronize()
        gradients = {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return float(total.detach().cpu()), gradients

    baseline_first_loss, baseline_first_grad = run_pass('blocking')
    baseline_repeat_loss, baseline_repeat_grad = run_pass('blocking')
    prefetch_loss, prefetch_grad = run_pass('prefetch')
    final_state_hash = state_sha256(model)
    baseline_noise = _gradient_difference(
        baseline_first_grad, baseline_repeat_grad)
    prefetch_difference = _gradient_difference(
        baseline_first_grad, prefetch_grad)
    result = {
        'status': 'passed' if (
            baseline_first_loss == baseline_repeat_loss == prefetch_loss
            and initial_state_hash == final_state_hash
            and prefetch_difference['relative_l2'] <= max(
                baseline_noise['relative_l2'] * 2.0, 1e-12)) else 'failed',
        'losses': {
            'baseline_first': baseline_first_loss,
            'baseline_repeat': baseline_repeat_loss,
            'prefetch': prefetch_loss,
        },
        'baseline_repeat_gradient_noise': baseline_noise,
        'prefetch_vs_baseline_gradient_difference': prefetch_difference,
        'model_state_unchanged': initial_state_hash == final_state_hash,
        **environment_receipt(),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if result['status'] != 'passed':
        raise RuntimeError(f'gradient-noise verification failed: {result}')


def combine_results(
        baseline_path, candidate_path, verification_path,
        gradient_noise_path, output_path):
    baseline = json.loads(Path(baseline_path).read_text(encoding='utf-8'))
    candidate = json.loads(Path(candidate_path).read_text(encoding='utf-8'))
    verification = json.loads(
        Path(verification_path).read_text(encoding='utf-8'))
    gradient_noise = json.loads(
        Path(gradient_noise_path).read_text(encoding='utf-8'))
    epoch_comparisons = []
    for base_epoch, candidate_epoch in zip(
            baseline['epochs'], candidate['epochs']):
        speedup = (
            base_epoch['seconds'] / candidate_epoch['seconds'] - 1.0) * 100.0
        epoch_comparisons.append({
            'epoch': base_epoch['epoch'],
            'baseline_seconds': base_epoch['seconds'],
            'candidate_seconds': candidate_epoch['seconds'],
            'candidate_throughput_gain_percent': speedup,
            'mean_loss_exactly_equal': (
                base_epoch['mean_loss'] == candidate_epoch['mean_loss']),
        })
    payload = {
        'status': 'completed',
        'experiment': 'V2-0004 training CUDA batch prefetch',
        'verification': verification,
        'gradient_noise_verification': gradient_noise,
        'baseline': baseline,
        'candidate': candidate,
        'comparison': {
            'initial_model_hash_equal': (
                baseline['initial_model_sha256'] ==
                candidate['initial_model_sha256']),
            'final_model_hash_equal': (
                baseline['final_model_sha256'] ==
                candidate['final_model_sha256']),
            'epoch_comparisons': epoch_comparisons,
            'candidate_stream_same_across_epochs': all(
                epoch['copy_stream_id'] ==
                candidate['persistent_copy_stream_id']
                for epoch in candidate['epochs']),
            'two_epoch_total_seconds': {
                'baseline': sum(
                    epoch['seconds'] for epoch in baseline['epochs']),
                'candidate': sum(
                    epoch['seconds'] for epoch in candidate['epochs']),
            },
            'prefetch_gradient_within_native_atomic_noise': (
                gradient_noise['status'] == 'passed'),
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--verify-real-transfer', action='store_true')
    group.add_argument('--smoke-candidate', action='store_true')
    group.add_argument('--verify-gradient-noise', action='store_true')
    group.add_argument('--arm', choices=['baseline', 'candidate'])
    group.add_argument('--combine', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline')
    parser.add_argument('--candidate')
    parser.add_argument('--verification')
    parser.add_argument('--gradient-noise')
    args = parser.parse_args()
    if args.verify_real_transfer:
        verify_real_transfer(args.output)
    elif args.smoke_candidate:
        smoke_candidate(args.output)
    elif args.verify_gradient_noise:
        verify_gradient_noise(args.output)
    elif args.arm:
        benchmark_arm(args.arm, args.output)
    else:
        if not all((args.baseline, args.candidate, args.verification,
                    args.gradient_noise)):
            parser.error(
                '--combine requires --baseline, --candidate, --verification '
                'and --gradient-noise')
        combine_results(
            args.baseline, args.candidate, args.verification,
            args.gradient_noise, args.output)


if __name__ == '__main__':
    main()
