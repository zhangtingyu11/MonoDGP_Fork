"""B1 microbenchmark for training-shaped pairwise rotated 3D IoU costs."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from lib.losses.asymmetric_interval_depth_loss import paired_iou3d


def _measure(query_count, target_count, layers, repeats, warmup, device):
    generator = torch.Generator(device=device).manual_seed(444 + target_count)
    prediction_center = torch.empty(
        query_count, 3, device=device).uniform_(-10, 60, generator=generator)
    prediction_center[:, 1].uniform_(-2, 2, generator=generator)
    prediction_dims = torch.empty(
        query_count, target_count, 3, device=device).uniform_(
            1, 5, generator=generator)
    prediction_yaw = torch.empty(
        query_count, device=device).uniform_(
            -3.14159265, 3.14159265, generator=generator)
    target_center = torch.empty(
        target_count, 3, device=device).uniform_(-10, 60, generator=generator)
    target_center[:, 1].uniform_(-2, 2, generator=generator)
    target_dims = torch.empty(
        target_count, 3, device=device).uniform_(1, 5, generator=generator)
    target_yaw = torch.empty(
        target_count, device=device).uniform_(
            -3.14159265, 3.14159265, generator=generator)

    query_index = torch.arange(
        query_count, device=device).repeat_interleave(target_count)
    target_index = torch.arange(
        target_count, device=device).repeat(query_count)

    def run():
        outputs = []
        for _ in range(layers):
            outputs.append(paired_iou3d(
                prediction_center[query_index],
                prediction_dims[query_index, target_index],
                prediction_yaw[query_index],
                target_center[target_index],
                target_dims[target_index],
                target_yaw[target_index]))
        return outputs

    for _ in range(warmup):
        run()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        outputs = run()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
        del outputs
    peak = torch.cuda.max_memory_allocated(device)
    values = torch.tensor(samples, dtype=torch.float64)
    return {
        "query_count": query_count,
        "target_count": target_count,
        "layers": layers,
        "pair_count_per_layer": query_count * target_count,
        "pair_count_total": query_count * target_count * layers,
        "milliseconds_mean": values.mean().item(),
        "milliseconds_median": values.median().item(),
        "milliseconds_min": values.min().item(),
        "milliseconds_max": values.max().item(),
        "incremental_peak_allocated_mib": (peak - baseline) / 2**20,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=550)
    parser.add_argument("--targets", type=int, nargs="+", default=(1, 3, 5, 10))
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    results = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "b1_training_shape": True,
        "results": [
            _measure(args.queries, count, args.layers,
                     args.repeats, args.warmup, device)
            for count in args.targets],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
