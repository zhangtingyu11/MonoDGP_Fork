#!/usr/bin/env python3
"""Decompose Exp47 AP headroom without training or changing its candidates.

The diagnostic keeps Exp47 E247's strict Car-only 50-query cache fixed and
separates three score/selection effects:

1. NMS traversal: use GT-assisted TP/IoU priority only while deciding which
   overlapping predictions survive, but retain Exp47 scores for official AP.
2. Within-image FP ordering: freeze baseline-NMS survivors and only permute the
   existing scores inside each image so unique TP@0.7 rows precede ignored rows
   and false positives.  Geometry and every image's score multiset stay fixed.
3. Cross-image calibration: on the same frozen survivors, permit the same score
   multiset to move across images.  Its incremental gain over (2) is the
   cross-image comparability headroom under this oracle construction.

All TP labels are score-independent Moderate-Car maximum-total-3D-IoU Hungarian
assignments after the repository's exact two-decimal KITTI quantization.  These
are descriptive GT oracles, not deployable methods or training targets.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torchvision
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.datasets.kitti.kitti_eval_python import kitti_common as kitti
from lib.datasets.kitti.kitti_eval_python.eval import (
    calculate_iou_partly,
    clean_data,
)
from lib.helpers.bev_nms_helper import _pairwise_bev_iou


CACHE = (
    ROOT
    / 'outputs/V2-0053_实验53_Exp47冻结TP07可靠性探针/val_cache.pt'
)
CHECKPOINT = (
    ROOT / 'outputs/V2-0047_实验47_去除三维IoU质量头/checkpoint_best.pth'
)
CONFIG = (
    ROOT / 'outputs/V2-0047_实验47_去除三维IoU质量头/resolved_config.yaml'
)
EXPECTED_NO_NMS_MODERATE = 24.252757700242608
IOU_THRESHOLD = 0.7
DIFFICULTY = 1
NMS_THRESHOLD = 0.8

APPROVED_ENVIRONMENT = {
    'sys_executable': str(ROOT / '.venv-cu129/bin/python'),
    'python': '3.10.20',
    'torch': '2.8.0+cu129',
    'torchvision': '0.23.0+cu129',
    'cuda': '12.9',
    'cudnn': 91002,
    'numba': '0.66.0',
    'numba_cuda': '0.30.4',
    'tf32_matmul': True,
    'tf32_cudnn': True,
    'deterministic_algorithms': True,
    'fill_uninitialized_memory': False,
    'cublas_workspace_config': ':4096:8',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    temporary.replace(path)


def runtime_snapshot() -> dict:
    import numba

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory = False
    snapshot = {
        'sys_executable': sys.executable,
        'python': platform.python_version(),
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'numba': numba.__version__,
        'numba_cuda': importlib.metadata.version('numba-cuda'),
        'tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
        'tf32_cudnn': bool(torch.backends.cudnn.allow_tf32),
        'deterministic_algorithms': (
            torch.are_deterministic_algorithms_enabled()),
        'fill_uninitialized_memory': (
            torch.utils.deterministic.fill_uninitialized_memory),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device': (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        'git_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_branch': subprocess.check_output(
            ['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        'git_status_short': subprocess.check_output(
            ['git', 'status', '--short'], cwd=ROOT, text=True).splitlines(),
    }
    mismatches = {
        key: {'expected': expected, 'actual': snapshot.get(key)}
        for key, expected in APPROVED_ENVIRONMENT.items()
        if snapshot.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f'approved runtime mismatch: {mismatches}')
    if not snapshot['cuda_available']:
        raise RuntimeError('host-visible CUDA is required by exact KITTI 3D IoU')
    return snapshot


def load_dataset() -> KITTI_Dataset:
    with CONFIG.open(encoding='utf-8') as stream:
        cfg = yaml.load(stream, Loader=yaml.Loader)
    dataset_cfg = copy.deepcopy(cfg['dataset'])
    dataset_cfg['batch_size'] = 1
    dataset = KITTI_Dataset(split='val', cfg=dataset_cfg)
    if dataset.data_augmentation:
        raise RuntimeError('validation dataset unexpectedly enables augmentation')
    return dataset


def load_cache(max_images: int = 0) -> dict:
    payload = torch.load(CACHE, map_location='cpu', weights_only=False)
    expected = (len(payload['image_ids']), 50)
    if tuple(payload['base_scores'].shape) != expected:
        raise RuntimeError('Exp47 cache score shape changed')
    if tuple(payload['detections'].shape) != (*expected, 14):
        raise RuntimeError('Exp47 cache detection shape changed')
    if len(set(payload['image_ids'].tolist())) != expected[0]:
        raise RuntimeError('Exp47 cache contains duplicate image IDs')
    if max_images:
        if max_images < 1 or max_images > expected[0]:
            raise ValueError('max-images must be within the cached split')
        for key in ('base_scores', 'detections', 'image_ids'):
            payload[key] = payload[key][:max_images]
    return payload


def cache_results(cache: dict) -> dict[int, np.ndarray]:
    results = {}
    for index, image_id in enumerate(cache['image_ids'].tolist()):
        rows = cache['detections'][index].numpy().copy()
        rows[:, 13] = cache['base_scores'][index].numpy()
        results[int(image_id)] = rows
    return results


def subset_dataset(dataset: KITTI_Dataset, image_ids) -> KITTI_Dataset:
    selected = copy.copy(dataset)
    selected.idx_list = [str(int(value)) for value in image_ids]
    return selected


def aligned_labels(dataset: KITTI_Dataset, results: dict[int, np.ndarray],
                   logger: logging.Logger) -> tuple[dict, dict, dict]:
    """Return score-independent Moderate Car labels and assigned 3D IoU."""
    image_ids = list(results)
    selected = subset_dataset(dataset, image_ids)
    dt_annos = selected._decoded_predictions_to_annos(results)
    gt_annos = kitti.get_label_annos(selected.label_dir, image_ids)
    overlaps, _, _, _ = calculate_iou_partly(
        gt_annos, dt_annos, metric=2, num_parts=min(50, len(image_ids)))
    labels = {}
    assigned = {}
    total_valid_gt = 0
    positive_count = ignored_count = negative_count = 0
    for image_id, gt, dt, overlap in zip(
            image_ids, gt_annos, dt_annos, overlaps):
        valid_gt_count, ignored_gt, ignored_dt, _ = clean_data(
            gt, dt, 0, DIFFICULTY)
        total_valid_gt += int(valid_gt_count)
        ignored_gt = np.asarray(ignored_gt)
        ignored_dt = np.asarray(ignored_dt)
        valid_gt = np.flatnonzero(ignored_gt == 0)
        ignored_gt_indices = np.flatnonzero(ignored_gt == 1)
        valid_det = np.flatnonzero(ignored_dt == 0)
        row_labels = np.zeros(len(dt['name']), dtype=np.int8)
        row_labels[ignored_dt != 0] = -1
        row_iou = np.zeros(len(dt['name']), dtype=np.float64)
        occupied = set()
        if len(valid_gt) and len(valid_det):
            gt_assignment, det_assignment = linear_sum_assignment(
                -overlap[np.ix_(valid_gt, valid_det)])
            for gt_local, det_local in zip(gt_assignment, det_assignment):
                gt_index = valid_gt[gt_local]
                det_index = valid_det[det_local]
                value = float(overlap[gt_index, det_index])
                row_iou[det_index] = value
                if value > IOU_THRESHOLD:
                    row_labels[det_index] = 1
                    occupied.add(int(det_index))
        remaining = np.asarray([
            value for value in valid_det if int(value) not in occupied
        ], dtype=np.int64)
        if len(ignored_gt_indices) and len(remaining):
            gt_assignment, det_assignment = linear_sum_assignment(
                -overlap[np.ix_(ignored_gt_indices, remaining)])
            for gt_local, det_local in zip(gt_assignment, det_assignment):
                gt_index = ignored_gt_indices[gt_local]
                det_index = remaining[det_local]
                value = float(overlap[gt_index, det_index])
                row_iou[det_index] = max(row_iou[det_index], value)
                if value > IOU_THRESHOLD:
                    row_labels[det_index] = -1
        labels[int(image_id)] = row_labels
        assigned[int(image_id)] = row_iou
        positive_count += int(np.sum(row_labels == 1))
        ignored_count += int(np.sum(row_labels == -1))
        negative_count += int(np.sum(row_labels == 0))
    summary = {
        'valid_gt_count': total_valid_gt,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'ignored_count': ignored_count,
        'qualified_gt_recall': (
            positive_count / total_valid_gt if total_valid_gt else 0.0),
    }
    logger.info('LABELS %s', json.dumps(summary, sort_keys=True))
    return labels, assigned, summary


def rank_nms(results: dict[int, np.ndarray], labels: dict, assigned: dict,
             threshold: float, use_oracle: bool) -> tuple[dict, dict]:
    variants = {}
    kept_indices = {}
    for image_id, raw_rows in results.items():
        rows = np.asarray(raw_rows)
        overlaps = _pairwise_bev_iou(rows)
        if use_oracle:
            order = sorted(
                range(len(rows)),
                key=lambda index: (
                    int(labels[image_id][index] == 1),
                    float(assigned[image_id][index]),
                    float(rows[index, 13]),
                    -index),
                reverse=True)
        else:
            order = sorted(
                range(len(rows)),
                key=lambda index: (float(rows[index, 13]), -index),
                reverse=True)
        kept = []
        while order:
            current = order.pop(0)
            kept.append(current)
            order = [
                candidate for candidate in order
                if (int(rows[current, 0]) != int(rows[candidate, 0])
                    or overlaps[current, candidate] <= threshold)
            ]
        # NMS traversal may use the oracle, but official AP always receives the
        # unchanged Exp47 confidence and its ordinary descending order.
        official = sorted(
            kept, key=lambda index: (float(rows[index, 13]), -index),
            reverse=True)
        variants[image_id] = rows[official].copy()
        kept_indices[image_id] = tuple(int(value) for value in kept)
    return variants, kept_indices


def _priority(label: int, original_score: float, stable_index: int):
    # Ignored detections do not contribute false positives; keep them between
    # positive and negative rows so the oracle only targets actual AP errors.
    category = 2 if label == 1 else (1 if label == -1 else 0)
    return category, original_score, -stable_index


def permute_existing_scores(results: dict[int, np.ndarray], labels: dict,
                            global_scope: bool) -> dict[int, np.ndarray]:
    """Permute, never synthesize, the existing score multiset."""
    output = {image_id: np.asarray(rows).copy()
              for image_id, rows in results.items()}
    if global_scope:
        references = [
            (image_id, index)
            for image_id, rows in output.items()
            for index in range(len(rows))
        ]
        score_pool = sorted(
            (float(output[image_id][index, 13])
             for image_id, index in references), reverse=True)
        priority_order = sorted(
            references,
            key=lambda item: _priority(
                int(labels[item[0]][item[1]]),
                float(output[item[0]][item[1], 13]),
                item[1]),
            reverse=True)
        for (image_id, index), score in zip(priority_order, score_pool):
            output[image_id][index, 13] = score
    else:
        for image_id, rows in output.items():
            score_pool = sorted((float(value) for value in rows[:, 13]),
                                reverse=True)
            priority_order = sorted(
                range(len(rows)),
                key=lambda index: _priority(
                    int(labels[image_id][index]),
                    float(rows[index, 13]), index),
                reverse=True)
            for index, score in zip(priority_order, score_pool):
                rows[index, 13] = score
    return output


def score_multiset(results: dict[int, np.ndarray]) -> np.ndarray:
    return np.sort(np.concatenate([
        np.asarray(rows)[:, 13].astype(np.float64, copy=False)
        for rows in results.values()
    ]))


def prediction_count(results: dict[int, np.ndarray]) -> int:
    return sum(len(rows) for rows in results.values())


def metric_receipt(receipt: dict) -> dict:
    return {
        'selection_score': float(receipt['selection_score']),
        'metrics': {key: float(value)
                    for key, value in receipt['metrics'].items()},
    }


def ap_triplet(receipt: dict) -> list[float]:
    metrics = receipt['metrics']
    return [
        float(metrics['Car_3d_easy_R40']),
        float(metrics['Car_3d_moderate_R40']),
        float(metrics['Car_3d_hard_R40']),
    ]


def evaluate(dataset: KITTI_Dataset, results: dict, logger: logging.Logger,
             name: str) -> dict:
    logger.info('EVALUATING_%s count=%d', name, prediction_count(results))
    receipt = metric_receipt(dataset.eval(
        results=results, logger=logger, return_metrics=True))
    logger.info('RESULT_%s triplet=%s', name, ap_triplet(receipt))
    return receipt


def quantized_score_summary(results: dict) -> dict:
    values = np.concatenate([
        np.asarray(rows)[:, 13] for rows in results.values()
    ])
    rounded = np.asarray([float(f'{value:.2f}') for value in values])
    return {
        'count': int(values.size),
        'finite': bool(np.isfinite(values).all()),
        'raw_unique': int(np.unique(values).size),
        'quantized_unique': int(np.unique(rounded).size),
        'quantized_zero_count': int(np.sum(rounded == 0.0)),
        'quantized_zero_fraction': float(np.mean(rounded == 0.0)),
        'raw_min': float(values.min()),
        'raw_max': float(values.max()),
    }


def run(args) -> dict:
    started = time.time()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('exp47_ap_attribution')
    runtime = runtime_snapshot()
    cache = load_cache(args.max_images)
    dataset = load_dataset()
    image_ids = [int(value) for value in cache['image_ids'].tolist()]
    selected_dataset = subset_dataset(dataset, image_ids)
    formal = args.max_images == 0

    manifest = {
        'started_at': datetime.now(timezone.utc).astimezone().isoformat(),
        'command': [sys.executable, *sys.argv],
        'working_directory': str(Path.cwd()),
        'formal': formal,
        'runtime': runtime,
        'inputs': {
            'cache': str(CACHE),
            'cache_sha256': sha256(CACHE),
            'checkpoint': str(CHECKPOINT),
            'checkpoint_sha256': sha256(CHECKPOINT),
            'config': str(CONFIG),
            'config_sha256': sha256(CONFIG),
        },
        'locked_controls': {
            'checkpoint_epoch': 247,
            'strict_car_only_query_count': 50,
            'top50_candidate_set_fixed': True,
            'iou_threshold': IOU_THRESHOLD,
            'difficulty': 'Moderate',
            'nms_threshold': args.nms_threshold,
            'geometry_rounded_to_repository_two_decimal_contract': True,
            'oracle_scores_are_permutations_of_existing_scores': True,
        },
    }
    atomic_json(output_dir / 'initial_manifest.json', manifest)

    baseline = cache_results(cache)
    pre_labels, pre_iou, pre_summary = aligned_labels(
        selected_dataset, baseline, logger)
    baseline_nms, baseline_keep = rank_nms(
        baseline, pre_labels, pre_iou, args.nms_threshold, use_oracle=False)
    oracle_nms, oracle_keep = rank_nms(
        baseline, pre_labels, pre_iou, args.nms_threshold, use_oracle=True)

    baseline_nms_labels, _, baseline_nms_summary = aligned_labels(
        selected_dataset, baseline_nms, logger)
    oracle_nms_labels, _, oracle_nms_summary = aligned_labels(
        selected_dataset, oracle_nms, logger)

    within_image = permute_existing_scores(
        baseline_nms, baseline_nms_labels, global_scope=False)
    global_scores = permute_existing_scores(
        baseline_nms, baseline_nms_labels, global_scope=True)
    combined = permute_existing_scores(
        oracle_nms, oracle_nms_labels, global_scope=True)
    no_nms_global = permute_existing_scores(
        baseline, pre_labels, global_scope=True)

    for name, original, changed in (
            ('within_image', baseline_nms, within_image),
            ('global', baseline_nms, global_scores),
            ('combined', oracle_nms, combined),
            ('no_nms_global', baseline, no_nms_global)):
        if not np.array_equal(score_multiset(original), score_multiset(changed)):
            raise RuntimeError(f'{name} changed the score multiset')

    evaluations = {
        'baseline_no_nms': evaluate(
            selected_dataset, baseline, logger, 'baseline_no_nms'),
        'baseline_nms080': evaluate(
            selected_dataset, baseline_nms, logger, 'baseline_nms080'),
        'local_nms_oracle_original_ap_scores': evaluate(
            selected_dataset, oracle_nms, logger,
            'local_nms_oracle_original_ap_scores'),
        'within_image_tp_fp_oracle_fixed_baseline_nms': evaluate(
            selected_dataset, within_image, logger,
            'within_image_tp_fp_oracle_fixed_baseline_nms'),
        'global_tp_fp_oracle_fixed_baseline_nms': evaluate(
            selected_dataset, global_scores, logger,
            'global_tp_fp_oracle_fixed_baseline_nms'),
        'combined_local_nms_and_global_tp_fp_oracle': evaluate(
            selected_dataset, combined, logger,
            'combined_local_nms_and_global_tp_fp_oracle'),
        'global_tp_fp_oracle_no_nms': evaluate(
            selected_dataset, no_nms_global, logger,
            'global_tp_fp_oracle_no_nms'),
    }
    if formal:
        actual = evaluations['baseline_no_nms']['selection_score']
        if abs(actual - EXPECTED_NO_NMS_MODERATE) > 1e-10:
            raise RuntimeError(
                f'baseline contract failed: {actual} != '
                f'{EXPECTED_NO_NMS_MODERATE}')

    baseline_nms_ap = evaluations['baseline_nms080']['selection_score']
    local_ap = evaluations[
        'local_nms_oracle_original_ap_scores']['selection_score']
    within_ap = evaluations[
        'within_image_tp_fp_oracle_fixed_baseline_nms']['selection_score']
    global_ap = evaluations[
        'global_tp_fp_oracle_fixed_baseline_nms']['selection_score']
    combined_ap = evaluations[
        'combined_local_nms_and_global_tp_fp_oracle']['selection_score']
    result = {
        'status': 'passed',
        'diagnostic': 'Exp47 E247 fixed-candidate AP attribution oracles',
        'formal': formal,
        'manifest': manifest,
        'protocol': {
            'baseline_no_nms': 'unchanged strict Car-only Exp47 cache',
            'baseline_nms080': 'ordinary Exp47 score traversal',
            'local_nms_oracle_original_ap_scores': (
                'GT TP/IoU changes NMS traversal only; AP scores unchanged'),
            'within_image_tp_fp_oracle_fixed_baseline_nms': (
                'baseline-NMS rows fixed; each image keeps its exact score '
                'multiset; TP then ignored then FP'),
            'global_tp_fp_oracle_fixed_baseline_nms': (
                'same baseline-NMS rows and global score multiset; scores may '
                'move between images; TP then ignored then FP'),
            'cross_image_contrast': (
                'global oracle minus within-image oracle; descriptive '
                'increment, not an additive causal effect'),
            'combined': (
                'oracle NMS traversal followed by global TP/FP score oracle'),
        },
        'counts': {
            'pre_nms': pre_summary,
            'baseline_nms': baseline_nms_summary,
            'oracle_nms': oracle_nms_summary,
            'prediction_pre_nms': prediction_count(baseline),
            'prediction_baseline_nms': prediction_count(baseline_nms),
            'prediction_oracle_nms': prediction_count(oracle_nms),
            'nms_keep_set_changed_images': int(sum(
                set(baseline_keep[key]) != set(oracle_keep[key])
                for key in baseline_keep)),
        },
        'score_quantization': quantized_score_summary(baseline),
        'evaluations': evaluations,
        'moderate_attribution': {
            'baseline_nms080': baseline_nms_ap,
            'local_nms_traversal_headroom': local_ap - baseline_nms_ap,
            'within_image_tp_fp_headroom': within_ap - baseline_nms_ap,
            'cross_image_increment_after_within_image_oracle': (
                global_ap - within_ap),
            'global_tp_fp_headroom': global_ap - baseline_nms_ap,
            'combined_headroom': combined_ap - baseline_nms_ap,
            'local_and_global_interaction': (
                combined_ap - local_ap - global_ap + baseline_nms_ap),
        },
        'limitations': [
            'Validation GT is used; every oracle is non-deployable.',
            'Moderate labels optimize only Moderate AP; Easy/Hard are descriptive.',
            'Hungarian TP identity is score-independent and need not equal the '
            'official evaluator assignment at every score threshold.',
            'The cross-image quantity is an incremental controlled contrast, '
            'not an independently identifiable causal effect.',
            'Two-decimal score quantization and ties are intentionally preserved.',
        ],
        'duration_seconds': time.time() - started,
        'finished_at': datetime.now(timezone.utc).astimezone().isoformat(),
    }
    atomic_json(output_dir / 'result.json', result)
    logger.info('FINAL_ATTRIBUTION %s', json.dumps(
        result['moderate_attribution'], sort_keys=True))
    print('EXP47_AP_ATTRIBUTION_OK ' + json.dumps(
        result['moderate_attribution'], ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--nms-threshold', type=float, default=NMS_THRESHOLD)
    args = parser.parse_args()
    if not 0.0 <= args.nms_threshold <= 1.0:
        raise ValueError('nms-threshold must be in [0, 1]')
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(output_dir / 'run.log'),
                  logging.StreamHandler(sys.stdout)])
    run(args)


if __name__ == '__main__':
    main()
