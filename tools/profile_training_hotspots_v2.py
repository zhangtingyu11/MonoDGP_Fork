#!/usr/bin/env python3
"""Fine-grained real-B16 training profiler for the cumulative MonoDGP stack."""

import argparse
import cProfile
import hashlib
import json
import math
from pathlib import Path
import pstats
import sqlite3
import statistics
import subprocess
import sys
import time
import types

import torch
import torchvision
from torch.profiler import ProfilerActivity, profile, record_function, schedule


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_config():
    from tools.benchmark_batched_same_image_matcher_v2 import (
        load_config as load_seq9_config,
    )

    cfg = load_seq9_config(True)
    cfg['trainer']['max_epoch'] = 1
    cfg['trainer']['save_path'] = '/tmp/monodgp_seq11_no_artifacts/'
    return cfg


def environment_receipt(command):
    import numba
    import numba_cuda

    script_path = Path(__file__).resolve()
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
        'profiler_script_sha256': hashlib.sha256(
            script_path.read_bytes()).hexdigest(),
        'gpu': torch.cuda.get_device_name(0),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'command': command,
    }


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return float('nan')
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def pearson(left, right):
    if len(left) != len(right) or len(left) < 2:
        return float('nan')
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_energy = sum((value - left_mean) ** 2 for value in left)
    right_energy = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_energy * right_energy)
    return numerator / denominator if denominator else 0.0


class PhaseEvents:
    def __init__(self):
        self.rows = []

    def measure(self, name):
        return _PhaseContext(self, name)

    def append_cpu(self, name, milliseconds):
        self.rows.append({
            'name': name,
            'cpu_wall_ms': milliseconds,
            'start': None,
            'end': None,
        })

    def resolve(self):
        resolved = []
        for row in self.rows:
            result = {
                'name': row['name'],
                'cpu_wall_ms': row['cpu_wall_ms'],
            }
            if row['start'] is not None:
                result['cuda_timeline_ms'] = row['start'].elapsed_time(
                    row['end'])
            resolved.append(result)
        return resolved


class _PhaseContext:
    def __init__(self, owner, name):
        self.owner = owner
        self.name = name
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.cpu_start = None

    def __enter__(self):
        self.cpu_start = time.perf_counter()
        self.start.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end.record()
        self.owner.rows.append({
            'name': self.name,
            'cpu_wall_ms': (time.perf_counter() - self.cpu_start) * 1000.0,
            'start': self.start,
            'end': self.end,
        })


def make_targets_and_dn(trainer, inputs, raw_targets):
    img_sizes = raw_targets['img_size']
    targets = trainer.prepare_targets(raw_targets, inputs.shape[0])
    dn_args = None
    if trainer.cfg['use_dn']:
        dn_args = (
            targets, trainer.cfg['scalar'],
            trainer.cfg['label_noise_scale'],
            trainer.cfg['box_noise_scale'],
            trainer.cfg['num_patterns'])
    return img_sizes, targets, dn_args


def weighted_total(criterion, losses):
    return sum(
        value * criterion.weight_dict[key]
        for key, value in losses.items()
        if key in criterion.weight_dict)


def force_training_log_sync(losses, weights):
    from utils import misc

    reduced = misc.reduce_dict(losses)
    return sum(
        float((value * weights[key]).item())
        for key, value in reduced.items()
        if key in weights)


def summarize_phase_rows(batch_rows, epoch_seconds):
    object_counts = [row['object_count'] for row in batch_rows]
    phase_names = list(batch_rows[0]['phases']) if batch_rows else []
    phases = {}
    for name in phase_names:
        cuda_values = [
            row['phases'][name].get('cuda_timeline_ms', 0.0)
            for row in batch_rows]
        cpu_values = [
            row['phases'][name]['cpu_wall_ms'] for row in batch_rows]
        phases[name] = {
            'cuda_timeline_ms': {
                'total': sum(cuda_values),
                'mean': statistics.fmean(cuda_values),
                'p50': percentile(cuda_values, 0.50),
                'p95': percentile(cuda_values, 0.95),
                'max': max(cuda_values),
                'correlation_with_object_count': pearson(
                    object_counts, cuda_values),
            },
            'cpu_wall_ms': {
                'total': sum(cpu_values),
                'mean': statistics.fmean(cpu_values),
                'p50': percentile(cpu_values, 0.50),
                'p95': percentile(cpu_values, 0.95),
                'max': max(cpu_values),
                'correlation_with_object_count': pearson(
                    object_counts, cpu_values),
            },
        }
    total_cuda_phase_ms = sum(
        value['cuda_timeline_ms']['total'] for value in phases.values())
    return {
        'epoch_seconds': epoch_seconds,
        'images_per_second': 3712 / epoch_seconds,
        'batches': len(batch_rows),
        'object_count': {
            'total': sum(object_counts),
            'mean': statistics.fmean(object_counts),
            'p50': percentile(object_counts, 0.50),
            'p95': percentile(object_counts, 0.95),
            'min': min(object_counts),
            'max': max(object_counts),
        },
        'phases': phases,
        'sum_of_phase_cuda_timeline_ms': total_cuda_phase_ms,
        'unattributed_epoch_ms': epoch_seconds * 1000.0 - total_cuda_phase_ms,
        'per_batch': batch_rows,
    }


def run_phase_profile(output_path):
    from lib.helpers.trainer_helper import CudaBatchPrefetcher
    from tools.benchmark_foreach_adamw_v2 import (
        BoardMemorySampler,
        build_stack,
        state_sha256,
    )

    cfg = load_config()
    train_loader, model, criterion, optimizer, trainer = build_stack(
        cfg, 'seq11-phase-profile')
    initial_hash = state_sha256(model)
    batch_source = CudaBatchPrefetcher(
        iter(train_loader), torch.device('cuda'),
        copy_stream=trainer.cuda_batch_copy_stream)
    model.train()
    all_events = []
    metadata = []
    loss_sum = 0.0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with BoardMemorySampler() as board:
        epoch_start = time.perf_counter()
        try:
            iterator = iter(batch_source)
            for batch_index in range(len(train_loader)):
                phases = PhaseEvents()
                wait_start = time.perf_counter()
                inputs, calibs, raw_targets, _ = next(iterator)
                phases.append_cpu(
                    'data_wait', (time.perf_counter() - wait_start) * 1000.0)
                with phases.measure('target_prepare'):
                    img_sizes, targets, dn_args = make_targets_and_dn(
                        trainer, inputs, raw_targets)
                object_count = sum(len(target['labels']) for target in targets)
                with phases.measure('zero_grad'):
                    optimizer.zero_grad()
                with phases.measure('model_forward'):
                    outputs = model(
                        inputs, calibs, targets, img_sizes, dn_args=dn_args)
                with phases.measure('matching_and_losses'):
                    losses = criterion(outputs, targets, None)
                with phases.measure('weighted_loss_sum'):
                    total = weighted_total(criterion, losses)
                with phases.measure('loss_logging_sync'):
                    logged_loss = force_training_log_sync(
                        losses, criterion.weight_dict)
                with phases.measure('backward'):
                    total.backward()
                with phases.measure('optimizer_step'):
                    optimizer.step()
                loss_sum += logged_loss
                all_events.append(phases)
                metadata.append({
                    'batch_index': batch_index,
                    'object_count': object_count,
                })
        finally:
            batch_source.close()
        torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - epoch_start
    batch_rows = []
    for meta, phase_events in zip(metadata, all_events):
        phase_map = {
            row['name']: row for row in phase_events.resolve()
        }
        batch_rows.append({**meta, 'phases': phase_map})
    result = {
        'status': 'completed',
        'mode': 'full_epoch_phase_profile',
        'batch_size': 16,
        'workers': train_loader.num_workers,
        'images': len(train_loader.dataset),
        'steps': len(train_loader),
        'initial_model_sha256': initial_hash,
        'mean_training_loss': loss_sum / len(train_loader),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'board_fb_peak_mib': board.peak_mib,
        'summary': summarize_phase_rows(batch_rows, epoch_seconds),
        **environment_receipt(
            f'{sys.executable} tools/profile_training_hotspots_v2.py '
            f'phases --output {output_path}'),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def wrap_record_function(module, label, attribute='forward'):
    original = getattr(module, attribute)

    def wrapped(_self, *args, **kwargs):
        with record_function(label):
            return original(*args, **kwargs)

    setattr(module, attribute, types.MethodType(wrapped, module))


def install_semantic_ranges(model, criterion):
    wrapped = set()

    def wrap_unique(module, label):
        if id(module) in wrapped:
            return
        wrapped.add(id(module))
        wrap_record_function(module, label)

    wrap_unique(model.backbone, 'module/backbone')
    for module in model.input_proj:
        wrap_unique(module, 'module/input_projection')
    wrap_unique(model.region_head, 'module/region_head')
    wrap_unique(model.depth_predictor, 'module/depth_predictor')
    wrap_unique(model.det2d_transformer, 'module/det2d_transformer')
    wrap_unique(model.det3d_transformer, 'module/det3d_transformer')
    for name in (
            'class_embed', 'bbox_embed', 'dim_embed_3d',
            'depth_embed', 'angle_embed'):
        for module in getattr(model, name):
            wrap_unique(module, f'module/prediction_head/{name}')
    wrap_unique(criterion.matcher, 'loss/matcher')

    original_get_loss = criterion.get_loss

    def wrapped_get_loss(_self, loss_name, *args, **kwargs):
        with record_function(f'loss/component/{loss_name}'):
            return original_get_loss(loss_name, *args, **kwargs)

    criterion.get_loss = types.MethodType(wrapped_get_loss, criterion)


def wrap_nvtx(module, label, attribute='forward'):
    original = getattr(module, attribute)

    def wrapped(_self, *args, **kwargs):
        with torch.cuda.nvtx.range(label):
            return original(*args, **kwargs)

    setattr(module, attribute, types.MethodType(wrapped, module))


def install_nvtx_ranges(model, criterion):
    wrapped = set()

    def wrap_unique(module, label):
        if id(module) in wrapped:
            return
        wrapped.add(id(module))
        wrap_nvtx(module, label)

    wrap_unique(model.backbone, 'module/backbone')
    wrap_unique(model.backbone[0], 'module/backbone/resnet')
    wrap_unique(model.backbone[1], 'module/backbone/position_embedding')
    if hasattr(model.backbone[0], 'body'):
        for name, module in model.backbone[0].body.named_children():
            if name in (
                    'conv1', 'bn1', 'relu', 'maxpool',
                    'layer1', 'layer2', 'layer3', 'layer4'):
                wrap_unique(module, f'module/backbone/{name}')
    for module in model.input_proj:
        wrap_unique(module, 'module/input_projection')
    wrap_unique(model.region_head, 'module/region_head')
    for name, module in model.region_head.named_modules():
        parts = name.split('.')
        if (name == 'upsample'
                or len(parts) == 2
                and parts[0] in ('input_proj', 'pred', 'attention')):
            wrap_unique(module, f'module/region_head/{name}')
    wrap_unique(model.depth_predictor, 'module/depth_predictor')
    for name, module in model.depth_predictor.named_children():
        wrap_unique(module, f'module/depth_predictor/{name}')
    wrap_unique(model.det2d_transformer, 'module/det2d_transformer')
    wrap_unique(model.det3d_transformer, 'module/det3d_transformer')
    for parent_name in ('det2d_transformer', 'det3d_transformer'):
        parent = getattr(model, parent_name)
        for name, module in parent.named_modules():
            if '.layers.' in name and name.rsplit('.', 1)[-1].isdigit():
                wrap_unique(module, f'module/{parent_name}/{name}')
            elif ('.layers.' in name and name.rsplit('.', 1)[-1] in (
                    'self_attn', 'cross_attn', 'cross_attn_depth',
                    'linear1', 'linear2')):
                wrap_unique(module, f'module/{parent_name}/{name}')
    wrap_unique(criterion.matcher, 'loss/matcher')


def execute_training_batch(
        trainer, model, criterion, optimizer, batch, use_nvtx=False):
    def region(name):
        return (torch.cuda.nvtx.range(name)
                if use_nvtx else _NullContext())

    inputs, calibs, raw_targets, _ = batch
    with region('phase/target_prepare'):
        img_sizes, targets, dn_args = make_targets_and_dn(
            trainer, inputs, raw_targets)
    with region('phase/zero_grad'):
        optimizer.zero_grad()
    with region('phase/model_forward'):
        outputs = model(inputs, calibs, targets, img_sizes, dn_args=dn_args)
    with region('phase/matching_and_losses'):
        losses = criterion(outputs, targets, None)
    with region('phase/weighted_loss_sum'):
        total = weighted_total(criterion, losses)
    with region('phase/loss_logging_sync'):
        logged_loss = force_training_log_sync(losses, criterion.weight_dict)
    with region('phase/backward'):
        total.backward()
    with region('phase/optimizer_step'):
        optimizer.step()
    return logged_loss, sum(len(target['labels']) for target in targets)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def run_nsys_timeline(output_path, warmup_steps, active_steps):
    from lib.helpers.trainer_helper import CudaBatchPrefetcher
    from tools.benchmark_foreach_adamw_v2 import build_stack, state_sha256

    cfg = load_config()
    train_loader, model, criterion, optimizer, trainer = build_stack(
        cfg, 'seq11-nsys-timeline')
    initial_hash = state_sha256(model)
    install_nvtx_ranges(model, criterion)
    batch_source = CudaBatchPrefetcher(
        iter(train_loader), torch.device('cuda'),
        copy_stream=trainer.cuda_batch_copy_stream)
    iterator = iter(batch_source)
    model.train()
    try:
        for _ in range(warmup_steps):
            execute_training_batch(
                trainer, model, criterion, optimizer, next(iterator))
        torch.cuda.synchronize()
        object_counts = []
        loss_sum = 0.0
        start = time.perf_counter()
        with torch.cuda.nvtx.range('profile/active_training_steps'):
            for _ in range(active_steps):
                with torch.cuda.nvtx.range('phase/data_wait'):
                    batch = next(iterator)
                logged_loss, object_count = execute_training_batch(
                    trainer, model, criterion, optimizer, batch,
                    use_nvtx=True)
                loss_sum += logged_loss
                object_counts.append(object_count)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
    finally:
        batch_source.close()
    result = {
        'status': 'completed',
        'mode': 'nsight_systems_timeline_workload',
        'warmup_steps': warmup_steps,
        'active_steps': active_steps,
        'batch_size': 16,
        'seconds': seconds,
        'milliseconds_per_batch': seconds * 1000.0 / active_steps,
        'mean_loss': loss_sum / active_steps,
        'object_counts': object_counts,
        'initial_model_sha256': initial_hash,
        **environment_receipt(
            f'{sys.executable} tools/profile_training_hotspots_v2.py '
            f'timeline --warmup {warmup_steps} --active {active_steps} '
            f'--output {output_path}'),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def run_data_profile(output_path, direct_samples):
    from lib.helpers.dataloader_helper import build_dataloader
    from lib.helpers.utils_helper import set_random_seed

    cfg = load_config()
    set_random_seed(cfg.get('random_seed', 444))
    train_loader, _ = build_dataloader(cfg['dataset'])
    arrivals = []
    images = 0
    iterator = iter(train_loader)
    previous = time.perf_counter()
    start = previous
    for inputs, _, _, _ in iterator:
        now = time.perf_counter()
        arrivals.append((now - previous) * 1000.0)
        previous = now
        images += int(inputs.shape[0])
    elapsed = time.perf_counter() - start

    dataset = train_loader.dataset
    profiler = cProfile.Profile()
    set_random_seed(cfg.get('random_seed', 444))
    profiler.enable()
    direct_start = time.perf_counter()
    for index in range(min(direct_samples, len(dataset))):
        dataset[index]
    direct_elapsed = time.perf_counter() - direct_start
    profiler.disable()
    stats = pstats.Stats(profiler)
    functions = []
    for (filename, line, name), values in stats.stats.items():
        primitive_calls, total_calls, total_time, cumulative_time, _ = values
        functions.append({
            'file': filename,
            'line': line,
            'function': name,
            'primitive_calls': primitive_calls,
            'total_calls': total_calls,
            'self_seconds': total_time,
            'cumulative_seconds': cumulative_time,
        })
    functions.sort(key=lambda row: row['cumulative_seconds'], reverse=True)
    result = {
        'status': 'completed',
        'mode': 'data_pipeline_profile',
        'batch_size': cfg['dataset']['batch_size'],
        'workers': train_loader.num_workers,
        'batches': len(arrivals),
        'images': images,
        'full_epoch_seconds': elapsed,
        'full_epoch_images_per_second': images / elapsed,
        'batch_arrival_ms': {
            'first': arrivals[0],
            'steady_mean': statistics.fmean(arrivals[1:]),
            'steady_p50': percentile(arrivals[1:], 0.50),
            'steady_p95': percentile(arrivals[1:], 0.95),
            'steady_max': max(arrivals[1:]),
        },
        'direct_dataset_samples': min(direct_samples, len(dataset)),
        'direct_dataset_seconds': direct_elapsed,
        'direct_samples_per_second': min(direct_samples, len(dataset)) / direct_elapsed,
        'top_profiled_functions_by_cumulative_time': functions[:100],
        **environment_receipt(
            f'{sys.executable} tools/profile_training_hotspots_v2.py '
            f'data --direct-samples {direct_samples} --output {output_path}'),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def run_saved_tensor_profile(output_path, warmup_steps):
    from lib.helpers.trainer_helper import CudaBatchPrefetcher
    from tools.benchmark_foreach_adamw_v2 import build_stack, state_sha256

    cfg = load_config()
    train_loader, model, criterion, optimizer, trainer = build_stack(
        cfg, 'seq11-saved-tensor-profile')
    initial_hash = state_sha256(model)
    batch_source = CudaBatchPrefetcher(
        iter(train_loader), torch.device('cuda'),
        copy_stream=trainer.cuda_batch_copy_stream)
    iterator = iter(batch_source)
    model.train()
    try:
        for _ in range(warmup_steps):
            execute_training_batch(
                trainer, model, criterion, optimizer, next(iterator))
        torch.cuda.synchronize()
        batch = next(iterator)
        inputs, calibs, raw_targets, _ = batch
        img_sizes, targets, dn_args = make_targets_and_dn(
            trainer, inputs, raw_targets)
        optimizer.zero_grad()
        torch.cuda.synchronize()
        memory_points = {
            'before_forward': {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
            },
        }

        module_stack = []
        handles = []
        module_names = {id(module): name or '<model>'
                        for name, module in model.named_modules()}

        def enter_module(module, _inputs):
            module_stack.append(module_names[id(module)])

        def leave_module(_module, _inputs, _output):
            module_stack.pop()

        for module in model.modules():
            handles.append(module.register_forward_pre_hook(enter_module))
            handles.append(module.register_forward_hook(leave_module))

        grouped = {}
        unique_storages = {}

        def pack(tensor):
            owner = module_stack[-1] if module_stack else '<functional-or-loss>'
            byte_count = tensor.numel() * tensor.element_size()
            row = grouped.setdefault(owner, {
                'saved_slots': 0, 'slot_bytes': 0,
                'cuda_slot_bytes': 0, 'cpu_slot_bytes': 0})
            row['saved_slots'] += 1
            row['slot_bytes'] += byte_count
            if tensor.is_cuda:
                row['cuda_slot_bytes'] += byte_count
            else:
                row['cpu_slot_bytes'] += byte_count
            storage = tensor.untyped_storage()
            key = (tensor.device.type, tensor.device.index,
                   storage.data_ptr(), storage.nbytes())
            unique_storages[key] = storage.nbytes()
            return tensor

        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                outputs = model(
                    inputs, calibs, targets, img_sizes, dn_args=dn_args)
                losses = criterion(outputs, targets, None)
                total = weighted_total(criterion, losses)
            torch.cuda.synchronize()
            memory_points['after_forward_and_loss'] = {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
            }
            total.backward()
            torch.cuda.synchronize()
            memory_points['after_backward'] = {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
            }
            optimizer.step()
            torch.cuda.synchronize()
            memory_points['after_optimizer'] = {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
            }
        finally:
            for handle in handles:
                handle.remove()
    finally:
        batch_source.close()
    owners = [
        {'owner': owner, **values}
        for owner, values in grouped.items()
    ]
    owners.sort(key=lambda row: row['cuda_slot_bytes'], reverse=True)
    result = {
        'status': 'completed',
        'mode': 'saved_tensor_and_memory_lifetime_profile',
        'batch_size': 16,
        'warmup_steps': warmup_steps,
        'initial_model_sha256': initial_hash,
        'object_count': sum(len(target['labels']) for target in targets),
        'memory_points_bytes': memory_points,
        'saved_tensor_slots': sum(row['saved_slots'] for row in owners),
        'saved_tensor_slot_bytes': sum(row['slot_bytes'] for row in owners),
        'saved_cuda_slot_bytes': sum(row['cuda_slot_bytes'] for row in owners),
        'unique_saved_storage_bytes': sum(unique_storages.values()),
        'by_innermost_forward_module': owners,
        'limitations': [
            'Repeated saved views of one storage are counted in slot bytes.',
            'Unique storage bytes deduplicate aliases but do not equal peak live memory.',
            'Functional operations and loss-side tensors are grouped together.',
        ],
        **environment_receipt(
            f'{sys.executable} tools/profile_training_hotspots_v2.py '
            f'memory --warmup {warmup_steps} --output {output_path}'),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def event_row(event):
    return {
        'key': event.key,
        'device_type': str(event.device_type),
        'device_index': getattr(event, 'device_index', None),
        'count': event.count,
        'self_cpu_time_us': event.self_cpu_time_total,
        'cpu_time_us': event.cpu_time_total,
        'self_device_time_us': event.self_device_time_total,
        'device_time_us': event.device_time_total,
        'self_cpu_memory_bytes': event.self_cpu_memory_usage,
        'cpu_memory_bytes': event.cpu_memory_usage,
        'self_device_memory_bytes': event.self_device_memory_usage,
        'device_memory_bytes': event.device_memory_usage,
        'input_shapes': event.input_shapes,
    }


def top_rows(events, field, count=150):
    return [event_row(event) for event in sorted(
    events, key=lambda item: getattr(item, field), reverse=True)[:count]]


def nearest_semantic_parent(event):
    parent = getattr(event, 'cpu_parent', None)
    while parent is not None:
        name = getattr(parent, 'name', '')
        if name.startswith(('phase/', 'module/', 'loss/')):
            return name
        parent = getattr(parent, 'cpu_parent', None)
    return '<unattributed>'


def synchronization_attribution(raw_events):
    grouped = {}
    for event in raw_events:
        name = getattr(event, 'name', '')
        if 'Synchronize' not in name and 'synchronize' not in name:
            continue
        parent = nearest_semantic_parent(event)
        key = (parent, name)
        row = grouped.setdefault(key, {
            'semantic_parent': parent,
            'operation': name,
            'count': 0,
            'self_cpu_time_us': 0.0,
            'example_stacks': [],
        })
        row['count'] += int(getattr(event, 'count', 1))
        row['self_cpu_time_us'] += float(event.self_cpu_time_total)
        stack = list(getattr(event, 'stack', ()) or ())
        if stack and stack not in row['example_stacks']:
            row['example_stacks'].append(stack)
            del row['example_stacks'][3:]
    return sorted(
        grouped.values(),
        key=lambda row: row['self_cpu_time_us'], reverse=True)


def run_operator_profile(output_path, wait_steps, warmup_steps, active_steps):
    from lib.helpers.trainer_helper import CudaBatchPrefetcher
    from tools.benchmark_foreach_adamw_v2 import build_stack, state_sha256

    cfg = load_config()
    train_loader, model, criterion, optimizer, trainer = build_stack(
        cfg, 'seq11-operator-profile')
    initial_hash = state_sha256(model)
    install_semantic_ranges(model, criterion)
    batch_source = CudaBatchPrefetcher(
        iter(train_loader), torch.device('cuda'),
        copy_stream=trainer.cuda_batch_copy_stream)
    total_steps = wait_steps + warmup_steps + active_steps
    model.train()
    object_counts = []
    profiler_schedule = schedule(
        wait=wait_steps, warmup=warmup_steps, active=active_steps, repeat=1)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
            activities=activities,
            schedule=profiler_schedule,
            record_shapes=True,
            profile_memory=True,
            with_stack=True) as prof:
        try:
            iterator = iter(batch_source)
            for _ in range(total_steps):
                with record_function('phase/data_wait'):
                    inputs, calibs, raw_targets, _ = next(iterator)
                with record_function('phase/target_prepare'):
                    img_sizes, targets, dn_args = make_targets_and_dn(
                        trainer, inputs, raw_targets)
                object_counts.append(sum(
                    len(target['labels']) for target in targets))
                with record_function('phase/zero_grad'):
                    optimizer.zero_grad()
                with record_function('phase/model_forward'):
                    outputs = model(
                        inputs, calibs, targets, img_sizes, dn_args=dn_args)
                with record_function('phase/matching_and_losses'):
                    losses = criterion(outputs, targets, None)
                with record_function('phase/weighted_loss_sum'):
                    total = weighted_total(criterion, losses)
                with record_function('phase/loss_logging_sync'):
                    force_training_log_sync(losses, criterion.weight_dict)
                with record_function('phase/backward'):
                    total.backward()
                with record_function('phase/optimizer_step'):
                    optimizer.step()
                prof.step()
        finally:
            batch_source.close()
    torch.cuda.synchronize()
    events = prof.key_averages(
        group_by_input_shape=False, group_by_stack_n=0)
    events_by_shape = prof.key_averages(
        group_by_input_shape=True, group_by_stack_n=0)
    semantic = [
        event_row(event) for event in events
        if event.key.startswith(('phase/', 'module/', 'loss/'))]
    result = {
        'status': 'completed',
        'mode': 'operator_profile',
        'batch_size': 16,
        'wait_steps': wait_steps,
        'warmup_steps': warmup_steps,
        'active_steps': active_steps,
        'total_steps_executed': total_steps,
        'object_counts_all_executed_steps': object_counts,
        'initial_model_sha256': initial_hash,
        'semantic_ranges': semantic,
        'synchronization_attribution': synchronization_attribution(
            prof.events()),
        'top_by_self_device_time': top_rows(
            events, 'self_device_time_total'),
        'top_by_self_device_time_and_shape': top_rows(
            events_by_shape, 'self_device_time_total', count=250),
        'top_by_self_cpu_time': top_rows(
            events, 'self_cpu_time_total'),
        'top_by_self_device_memory': top_rows(
            events, 'self_device_memory_usage'),
        **environment_receipt(
            f'{sys.executable} tools/profile_training_hotspots_v2.py ops '
            f'--wait {wait_steps} --warmup {warmup_steps} '
            f'--active {active_steps} --output {output_path}'),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def summarize(phase_path, operator_path, output_path):
    phase_result = json.loads(Path(phase_path).read_text(encoding='utf-8'))
    operator_result = json.loads(
        Path(operator_path).read_text(encoding='utf-8'))
    if phase_result['initial_model_sha256'] != operator_result[
            'initial_model_sha256']:
        raise RuntimeError('profiler runs used different initial model states')
    result = {
        'status': 'completed',
        'experiment': 'V2-0011 fine-grained training hotspot profile',
        'full_epoch_phase_profile': phase_result,
        'operator_profile': operator_result,
        'limitations': [
            'The full-epoch CUDA-event run is the source of absolute timing.',
            'The operator profiler adds overhead and is used only for hotspot proportions and attribution.',
            'Module device times are inclusive and nested ranges must not be added together.',
            'Only one full epoch was timed, so run-to-run variance is not estimated.',
        ],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def _nsys_active_bounds(connection):
    row = connection.execute(
        """SELECT n.start, n.end FROM NVTX_EVENTS n
           LEFT JOIN StringIds s ON s.id=n.textId
           WHERE COALESCE(n.text,s.value)='profile/active_training_steps'""").fetchone()
    if row is None:
        raise RuntimeError('Nsight database has no active training range')
    return row


def _nsys_timeline_summary(database_path, active_steps):
    connection = sqlite3.connect(database_path)
    try:
        start, end = _nsys_active_bounds(connection)
        streams = connection.execute(
            """SELECT streamId, COUNT(*), SUM(end-start)
               FROM CUPTI_ACTIVITY_KIND_KERNEL
               WHERE start>=? AND start<? GROUP BY streamId
               ORDER BY SUM(end-start) DESC""", (start, end)).fetchall()
        main_stream, kernel_count, kernel_ns = streams[0]
        copy_row = connection.execute(
            """SELECT COUNT(*), COALESCE(SUM(end-start),0),
                      COALESCE(SUM(bytes),0)
               FROM CUPTI_ACTIVITY_KIND_MEMCPY
               WHERE streamId=? AND start>=? AND start<?""",
            (main_stream, start, end)).fetchone()
        memset_row = connection.execute(
            """SELECT COUNT(*), COALESCE(SUM(end-start),0)
               FROM CUPTI_ACTIVITY_KIND_MEMSET
               WHERE streamId=? AND start>=? AND start<?""",
            (main_stream, start, end)).fetchone()
        intervals = connection.execute(
            """SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL
               WHERE streamId=? AND start>=? AND start<?
               UNION ALL SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY
               WHERE streamId=? AND start>=? AND start<?
               UNION ALL SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET
               WHERE streamId=? AND start>=? AND start<? ORDER BY start""",
            (main_stream, start, end, main_stream, start, end,
             main_stream, start, end)).fetchall()
        gap_ns = sum(max(0, current[0] - previous[1])
                     for previous, current in zip(intervals, intervals[1:]))
        kernel_durations = [row[0] for row in connection.execute(
            """SELECT end-start FROM CUPTI_ACTIVITY_KIND_KERNEL
               WHERE start>=? AND start<?""", (start, end))]
        copy_streams = connection.execute(
            """SELECT streamId, COUNT(*), SUM(end-start), SUM(bytes)
               FROM CUPTI_ACTIVITY_KIND_MEMCPY
               WHERE start>=? AND start<? AND streamId<>?
               GROUP BY streamId ORDER BY SUM(end-start) DESC""",
            (start, end, main_stream)).fetchall()
        return {
            'active_steps': active_steps,
            'active_range_ms_per_batch': (end - start) / 1e6 / active_steps,
            'main_stream': int(main_stream),
            'main_stream_kernel_count_per_batch': kernel_count / active_steps,
            'main_stream_kernel_ms_per_batch': kernel_ns / 1e6 / active_steps,
            'main_stream_copy_ms_per_batch': copy_row[1] / 1e6 / active_steps,
            'main_stream_memset_ms_per_batch': memset_row[1] / 1e6 / active_steps,
            'main_stream_gap_ms_per_batch': gap_ns / 1e6 / active_steps,
            'main_stream_gap_fraction': gap_ns / (end - start),
            'all_kernel_count': len(kernel_durations),
            'kernel_under_10us_count': sum(value < 10_000 for value in kernel_durations),
            'kernel_under_10us_fraction': (
                sum(value < 10_000 for value in kernel_durations)
                / len(kernel_durations)),
            'kernel_under_50us_count': sum(value < 50_000 for value in kernel_durations),
            'kernel_under_50us_fraction': (
                sum(value < 50_000 for value in kernel_durations)
                / len(kernel_durations)),
            'non_main_copy_streams': [
                {'stream': row[0], 'count': row[1],
                 'ms_per_batch': row[2] / 1e6 / active_steps,
                 'mib_per_batch': row[3] / 1048576.0 / active_steps}
                for row in copy_streams],
        }
    finally:
        connection.close()


def _nsys_module_summary(database_path, active_steps):
    connection = sqlite3.connect(database_path)
    try:
        start, end = _nsys_active_bounds(connection)
        rows = connection.execute(
            """WITH ranges AS (
                 SELECT n.start,n.end,COALESCE(n.text,s.value) name,n.globalTid
                 FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId
                 WHERE COALESCE(n.text,s.value) LIKE 'module/%'
                   AND n.start>=? AND n.start<?),
               ops AS (
                 SELECT r.start launch,k.end-k.start duration,r.globalTid
                 FROM CUPTI_ACTIVITY_KIND_KERNEL k
                 JOIN CUPTI_ACTIVITY_KIND_RUNTIME r USING(correlationId)
                 UNION ALL
                 SELECT r.start launch,m.end-m.start duration,r.globalTid
                 FROM CUPTI_ACTIVITY_KIND_MEMCPY m
                 JOIN CUPTI_ACTIVITY_KIND_RUNTIME r USING(correlationId)
                 UNION ALL
                 SELECT r.start launch,m.end-m.start duration,r.globalTid
                 FROM CUPTI_ACTIVITY_KIND_MEMSET m
                 JOIN CUPTI_ACTIVITY_KIND_RUNTIME r USING(correlationId))
               SELECT ranges.name,COUNT(*),SUM(ops.duration)
               FROM ops JOIN ranges ON ops.globalTid=ranges.globalTid
                 AND ops.launch BETWEEN ranges.start AND ranges.end
               GROUP BY ranges.name ORDER BY SUM(ops.duration) DESC""",
            (start, end)).fetchall()
        return [
            {'module': name, 'gpu_ops': count,
             'device_ms_per_batch': duration / 1e6 / active_steps}
            for name, count, duration in rows]
    finally:
        connection.close()


def extend_summary(args):
    result = json.loads(Path(args.output).read_text(encoding='utf-8'))
    data = json.loads(Path(args.data).read_text(encoding='utf-8'))
    memory = json.loads(Path(args.memory).read_text(encoding='utf-8'))
    timeline_workload = json.loads(
        Path(args.timeline_workload).read_text(encoding='utf-8'))
    deep_workload = json.loads(
        Path(args.deep_workload).read_text(encoding='utf-8'))
    expected_hash = result['full_epoch_phase_profile']['initial_model_sha256']
    for receipt in (data, memory, timeline_workload, deep_workload):
        if receipt.get('initial_model_sha256', expected_hash) != expected_hash:
            raise RuntimeError('extended profiler run used a different model state')
    result['extended_analysis'] = {
        'clean_nsys_timeline': {
            **_nsys_timeline_summary(args.timeline_db, 20),
            'workload': timeline_workload,
        },
        'deep_forward_modules': {
            'workload': deep_workload,
            'modules': _nsys_module_summary(args.deep_db, 10),
        },
        'data_pipeline': data,
        'saved_tensors_and_memory_lifetime': memory,
        'hardware_counter_attempt': {
            'status': 'blocked',
            'tool': 'Nsight Compute 2025.2.1',
            'error': 'ERR_NVGPUCTRPERM',
            'meaning': 'The current user cannot read NVIDIA GPU performance counters.',
        },
        'correctness_reruns': [
            {
                'failed_attempt': 'Nsight Systems default CUDA Event tracing warned about false dependencies.',
                'fix': 'Disabled cuda-event-trace without changing model, data, or batch scope.',
                'rerun_count': 1,
            },
        ],
    }
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode', choices=(
            'phases', 'ops', 'timeline', 'data', 'memory',
            'summarize', 'extend'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--wait', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--active', type=int, default=12)
    parser.add_argument('--phases')
    parser.add_argument('--operators')
    parser.add_argument('--direct-samples', type=int, default=256)
    parser.add_argument('--data')
    parser.add_argument('--memory')
    parser.add_argument('--timeline-db')
    parser.add_argument('--timeline-workload')
    parser.add_argument('--deep-db')
    parser.add_argument('--deep-workload')
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.mode == 'phases':
        run_phase_profile(args.output)
    elif args.mode == 'ops':
        run_operator_profile(
            args.output, args.wait, args.warmup, args.active)
    elif args.mode == 'timeline':
        run_nsys_timeline(args.output, args.warmup, args.active)
    elif args.mode == 'data':
        run_data_profile(args.output, args.direct_samples)
    elif args.mode == 'memory':
        run_saved_tensor_profile(args.output, args.warmup)
    elif args.mode == 'extend':
        extend_summary(args)
    else:
        summarize(args.phases, args.operators, args.output)


if __name__ == '__main__':
    main()
