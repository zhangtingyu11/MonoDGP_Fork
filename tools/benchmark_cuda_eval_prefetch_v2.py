#!/usr/bin/env python3
"""Full KITTI correctness and performance checks for validation prefetch."""

import argparse
import collections
import gc
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.datasets.kitti.kitti_dataset import KITTI_Dataset  # noqa: E402
from lib.helpers.dataloader_helper import my_worker_init_fn  # noqa: E402
from lib.helpers.model_helper import build_model  # noqa: E402
from lib.helpers.tester_helper import CudaEvalBatchPrefetcher, Tester  # noqa: E402
from lib.helpers.utils_helper import set_random_seed  # noqa: E402
import lib.models.monodgp.backbone as backbone_module  # noqa: E402


def load_config():
    with (ROOT / 'configs/monodgp.yaml').open(encoding='utf-8') as handle:
        cfg = yaml.load(handle, Loader=yaml.Loader)
    cfg['dataset']['batch_size'] = 16
    cfg['tester']['export_predictions'] = False
    return cfg


def build_val_loader(dataset_cfg, pin_memory):
    cfg = dict(dataset_cfg)
    dataset = KITTI_Dataset(split=cfg['test_split'], cfg=cfg)
    return DataLoader(
        dataset=dataset, batch_size=16, num_workers=4,
        worker_init_fn=my_worker_init_fn, shuffle=False,
        pin_memory=pin_memory, drop_last=False)


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


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def update_tree_digest(digest, value, prefix='root'):
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(f'{prefix}|tensor|{tensor.dtype}|{tuple(tensor.shape)}'.encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(f'{prefix}|array|{array.dtype}|{array.shape}'.encode())
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        digest.update(f'{prefix}|dict|{len(value)}'.encode())
        for key in sorted(value):
            update_tree_digest(digest, value[key], f'{prefix}/{key}')
    elif isinstance(value, (tuple, list)):
        digest.update(f'{prefix}|sequence|{len(value)}'.encode())
        for index, item in enumerate(value):
            update_tree_digest(digest, item, f'{prefix}/{index}')
    else:
        digest.update(f'{prefix}|value|{type(value).__name__}|{value!r}'.encode())


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


def verify_full_transfer(output_path):
    cfg = load_config()
    seed = cfg.get('random_seed', 444)

    set_random_seed(seed)
    ordinary_loader = build_val_loader(cfg['dataset'], pin_memory=False)
    ordinary_tree = hashlib.sha256()
    ordinary_order = hashlib.sha256()
    ordinary_batches = 0
    ordinary_images = 0
    for batch in ordinary_loader:
        update_tree_digest(ordinary_tree, batch)
        image_ids = torch.as_tensor(batch[3]['img_id'])
        ordinary_order.update(image_ids.numpy().tobytes())
        ordinary_images += image_ids.numel()
        ordinary_batches += 1

    set_random_seed(seed)
    prefetch_loader = build_val_loader(cfg['dataset'], pin_memory=True)
    device = torch.device('cuda')
    copy_stream = torch.cuda.Stream(device=device)
    stream_id = copy_stream.cuda_stream
    tapped = TappedIterator(iter(prefetch_loader))
    prefetcher = CudaEvalBatchPrefetcher(
        tapped, device, copy_stream=copy_stream)
    prefetch_tree = hashlib.sha256()
    prefetch_order = hashlib.sha256()
    copied_elements = 0
    mismatched_elements = 0
    prefetch_batches = 0
    prefetch_images = 0
    unpinned_sources = 0
    try:
        for gpu_batch in prefetcher:
            host_batch = tapped.pending.popleft()
            update_tree_digest(prefetch_tree, host_batch)
            image_ids = torch.as_tensor(host_batch[3]['img_id'])
            prefetch_order.update(image_ids.numpy().tobytes())
            prefetch_images += image_ids.numel()
            prefetch_batches += 1

            host_sources = (
                host_batch[0], host_batch[1], host_batch[3]['img_size'])
            gpu_sources = (gpu_batch[0], gpu_batch[1], gpu_batch[4])
            for host, gpu in zip(host_sources, gpu_sources):
                if not host.is_pinned():
                    unpinned_sources += 1
                restored = gpu.detach().cpu()
                copied_elements += host.numel()
                equal = host == restored
                if host.dtype.is_floating_point:
                    equal |= torch.isnan(host) & torch.isnan(restored)
                mismatched_elements += int((~equal).sum().item())
    finally:
        prefetcher.close()

    result = {
        'status': 'passed' if (
            ordinary_batches == prefetch_batches == len(prefetch_loader)
            and ordinary_images == prefetch_images == len(prefetch_loader.dataset)
            and ordinary_order.digest() == prefetch_order.digest()
            and ordinary_tree.digest() == prefetch_tree.digest()
            and mismatched_elements == 0 and unpinned_sources == 0
            and copy_stream.cuda_stream == stream_id) else 'failed',
        'batch_size': 16,
        'ordinary_batches': ordinary_batches,
        'prefetch_batches': prefetch_batches,
        'ordinary_images': ordinary_images,
        'prefetch_images': prefetch_images,
        'ordinary_order_sha256': ordinary_order.hexdigest(),
        'prefetch_order_sha256': prefetch_order.hexdigest(),
        'ordinary_full_batch_sha256': ordinary_tree.hexdigest(),
        'prefetch_full_batch_sha256': prefetch_tree.hexdigest(),
        'copied_tensor_elements': copied_elements,
        'mismatched_copied_elements': mismatched_elements,
        'unpinned_source_tensors': unpinned_sources,
        'copy_stream_reused': copy_stream.cuda_stream == stream_id,
        **environment_receipt(),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if result['status'] != 'passed':
        raise RuntimeError(f'full validation transfer verification failed: {result}')


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


def prediction_sha256(results):
    digest = hashlib.sha256()
    detections = 0
    for image_id in sorted(results):
        array = np.ascontiguousarray(results[image_id])
        digest.update(str(int(image_id)).encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
        detections += len(array)
    return digest.hexdigest(), detections


def make_tester(cfg, model, loader, use_prefetch, logger):
    train_cfg = {
        'save_path': '/tmp/monodgp_seq5_no_artifacts/',
        'save_all': False,
        'use_cuda_eval_prefetch': use_prefetch,
    }
    tester_cfg = dict(cfg['tester'])
    tester_cfg['export_predictions'] = False
    return Tester(
        cfg=tester_cfg, model=model, dataloader=loader, logger=logger,
        train_cfg=train_cfg,
        model_name=('seq5_prefetch' if use_prefetch else 'seq5_control'))


def warm_one_batch(model, loader, device):
    inputs, calibs, targets, info = next(iter(loader))
    with torch.inference_mode():
        model(
            inputs.to(device), calibs.to(device), targets,
            info['img_size'].to(device), dn_args=0)
    torch.cuda.synchronize(device)


def benchmark(output_path, checkpoint_path):
    cfg = load_config()
    set_random_seed(cfg.get('random_seed', 444))
    control_loader = build_val_loader(cfg['dataset'], pin_memory=False)
    candidate_loader = build_val_loader(cfg['dataset'], pin_memory=True)

    backbone_module.is_main_process = lambda: False
    model, _ = build_model(cfg['model'])
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    device = torch.device('cuda')
    model = model.to(device).eval()

    logger = logging.getLogger('seq5-cuda-eval-prefetch')
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    control = make_tester(cfg, model, control_loader, False, logger)
    candidate = make_tester(cfg, model, candidate_loader, True, logger)

    warm_one_batch(model, control_loader, device)
    warm_one_batch(model, candidate_loader, device)
    runs = []
    for position, arm in enumerate(
            ('control', 'candidate', 'candidate', 'control'), start=1):
        tester = control if arm == 'control' else candidate
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        with BoardMemorySampler() as board:
            started = time.perf_counter()
            results = tester.inference()
            torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - started
            prediction_hash, detections = prediction_sha256(results)
            ap_started = time.perf_counter()
            moderate_ap = tester.evaluate(results)
            torch.cuda.synchronize(device)
            ap_seconds = time.perf_counter() - ap_started
        runs.append({
            'position': position,
            'arm': arm,
            'images': len(results),
            'detections': detections,
            'prediction_sha256': prediction_hash,
            'moderate_3d_ap_r40': float(moderate_ap),
            'inference_seconds': inference_seconds,
            'images_per_second': len(results) / inference_seconds,
            'ap_seconds': ap_seconds,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'board_fb_peak_mib': board.peak_mib,
            'copy_stream_id': (
                candidate.cuda_eval_copy_stream.cuda_stream
                if arm == 'candidate' else None),
        })
        del results
        gc.collect()

    prediction_hashes = {run['prediction_sha256'] for run in runs}
    ap_values = {run['moderate_3d_ap_r40'] for run in runs}
    control_seconds = [
        run['inference_seconds'] for run in runs if run['arm'] == 'control']
    candidate_seconds = [
        run['inference_seconds'] for run in runs if run['arm'] == 'candidate']
    control_mean = sum(control_seconds) / len(control_seconds)
    candidate_mean = sum(candidate_seconds) / len(candidate_seconds)
    result = {
        'status': 'passed' if (
            len(prediction_hashes) == 1 and len(ap_values) == 1
            and all(run['images'] == len(control_loader.dataset) for run in runs)
            and len({run['copy_stream_id'] for run in runs
                     if run['arm'] == 'candidate'}) == 1) else 'failed',
        'design': {
            'order': ['control', 'candidate', 'candidate', 'control'],
            'batch_size': 16,
            'workers': 4,
            'checkpoint': str(checkpoint_path),
            'checkpoint_sha256': file_sha256(checkpoint_path),
            'checkpoint_epoch': checkpoint.get('epoch'),
        },
        'runs': runs,
        'comparison': {
            'predictions_exactly_equal': len(prediction_hashes) == 1,
            'prediction_sha256': next(iter(prediction_hashes)),
            'ap_exactly_equal': len(ap_values) == 1,
            'moderate_3d_ap_r40': next(iter(ap_values)),
            'control_mean_inference_seconds': control_mean,
            'candidate_mean_inference_seconds': candidate_mean,
            'candidate_time_reduction_percent': (
                (control_mean - candidate_mean) / control_mean * 100.0),
            'candidate_throughput_gain_percent': (
                (control_mean / candidate_mean - 1.0) * 100.0),
        },
        **environment_receipt(),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if result['status'] != 'passed':
        raise RuntimeError(f'validation benchmark correctness failed: {result}')


def combine(verification_path, benchmark_path, output_path):
    verification = json.loads(
        Path(verification_path).read_text(encoding='utf-8'))
    performance = json.loads(
        Path(benchmark_path).read_text(encoding='utf-8'))
    result = {
        'status': ('completed' if verification['status'] == 'passed'
                   and performance['status'] == 'passed' else 'failed'),
        'experiment': 'V2-0005 validation CUDA batch prefetch',
        'verification': verification,
        'performance': performance,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if result['status'] != 'completed':
        raise RuntimeError(f'cannot combine failed results: {result}')


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--verify-transfer', action='store_true')
    group.add_argument('--benchmark', action='store_true')
    group.add_argument('--combine', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--checkpoint', default='checkpoints/monodgp_val_22.4953.pth')
    parser.add_argument('--verification')
    parser.add_argument('--performance')
    args = parser.parse_args()

    if args.verify_transfer:
        verify_full_transfer(args.output)
    elif args.benchmark:
        benchmark(args.output, args.checkpoint)
    else:
        if not args.verification or not args.performance:
            parser.error('--combine requires --verification and --performance')
        combine(args.verification, args.performance, args.output)


if __name__ == '__main__':
    main()
