"""Read-only contract and timing for gradient monitoring on one real B16."""

import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.gradient_monitor import GradientMonitor  # noqa: E402
from tools.benchmark_foreach_adamw_v2 import build_stack  # noqa: E402


def timed(callable_):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    value = callable_()
    end.record()
    end.synchronize()
    return value, start.elapsed_time(end)


def main():
    from lib.helpers.config_helper import load_config

    cfg = load_config(ROOT / 'configs/monodgp_exp45.yaml')
    cfg['dataset']['batch_size'] = 16
    cfg['trainer']['save_frequency'] = 1000
    cfg['trainer']['swanlab']['enabled'] = False
    strict = bool(cfg['trainer'].get('strict_determinism', False))
    torch.use_deterministic_algorithms(strict, warn_only=False)
    if strict:
        from lib.models.monodgp.ops.functions.ms_deform_attn_func import (
            ensure_deterministic_msda_available,
        )
        ensure_deterministic_msda_available()
    train_loader, model, criterion, _, trainer = build_stack(
        cfg, 'gradient-monitor-b16')
    inputs, calibs, raw_targets, _ = next(iter(train_loader))
    inputs = inputs.cuda(non_blocking=True)
    calibs = calibs.cuda(non_blocking=True)
    raw_targets = {
        key: value.cuda(non_blocking=True)
        for key, value in raw_targets.items()}
    targets = trainer.prepare_targets(raw_targets, inputs.shape[0])
    model.train()
    criterion.train()
    model.zero_grad(set_to_none=True)
    outputs = model(
        inputs, calibs, targets, raw_targets['img_size'], dn_args=0)
    losses = criterion(outputs, targets, None)
    total = sum(
        value * criterion.weight_dict[key]
        for key, value in losses.items()
        if key in criterion.weight_dict)
    total.backward()
    torch.cuda.synchronize()

    missing_gradients = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None]

    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad]
    versions_before = {
        id(parameter.grad): parameter.grad._version
        for parameter in parameters if parameter.grad is not None}
    monitor = GradientMonitor(model, module_interval=30)
    monitor.observe(1)
    monitor.observe(0)
    versions_after = {
        id(parameter.grad): parameter.grad._version
        for parameter in parameters if parameter.grad is not None}
    if versions_before != versions_after:
        raise RuntimeError('gradient monitor mutated a gradient tensor')
    summary = monitor.finalize()
    if summary['observed_batch_count'] != 2:
        raise RuntimeError('gradient epoch summary lost observed batches')
    if summary['nonfinite_batch_count'] != 0:
        raise RuntimeError('non-finite gradients in the B16 contract batch')

    for _ in range(3):
        GradientMonitor(model, 30).observe(1)
        GradientMonitor(model, 30).observe(0)
    global_ms, module_ms = [], []
    for _ in range(20):
        _, elapsed = timed(lambda: GradientMonitor(model, 30).observe(1))
        global_ms.append(elapsed)
        snapshot, elapsed = timed(
            lambda: GradientMonitor(model, 30).observe(0))
        module_ms.append(elapsed)
    global_median = statistics.median(global_ms)
    module_median = statistics.median(module_ms)
    module_snapshots = 9
    estimated_epoch_ms = (
        global_median * 232
        + max(module_median - global_median, 0.0) * module_snapshots)
    print(f'global_only_ms={global_ms}')
    print(f'module_snapshot_ms={module_ms}')
    print(f'global_median_ms={global_median:.6f}')
    print(f'module_snapshot_median_ms={module_median:.6f}')
    print(f'estimated_epoch_seconds={estimated_epoch_ms / 1000.0:.6f}')
    print(f'module_metric_count={len(snapshot)}')
    print(f'missing_gradients={missing_gradients}')
    print(f'epoch_summary={summary}')
    print('GRADIENT_MONITOR_READ_ONLY_CONTRACT_OK')


if __name__ == '__main__':
    main()
