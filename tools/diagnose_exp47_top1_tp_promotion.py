#!/usr/bin/env python3
"""Promote at most one TP per image to a strict top score for Exp47 E247."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import diagnose_exp47_ap_attribution as base


EXPECTED_NO_NMS_MODERATE = 24.252757700242608


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


def quantized_score(value: float) -> float:
    return float(f'{float(value):.2f}')


def promote_one_tp_per_image(results: dict[int, np.ndarray], labels: dict):
    """Make one existing unique TP the strict top score in each eligible image.

    The selected row is the highest-original-score TP.  Its score changes only
    when it is not already the unique top row after the repository's exact
    two-decimal score quantization.  The new value is the smallest 0.01 grid
    point strictly above the image's current quantized maximum.
    """
    promoted = {
        image_id: np.asarray(rows).copy()
        for image_id, rows in results.items()
    }
    changed_rows = {}
    summary = {
        'image_count': len(results),
        'image_without_unique_tp_candidate': 0,
        'image_with_unique_tp_candidate': 0,
        'image_already_unique_top_tp': 0,
        'image_promoted': 0,
    }
    for image_id, rows in promoted.items():
        row_labels = np.asarray(labels[image_id])
        positives = np.flatnonzero(row_labels == 1)
        if len(positives) == 0:
            summary['image_without_unique_tp_candidate'] += 1
            continue
        summary['image_with_unique_tp_candidate'] += 1
        original_scores = rows[:, 13].astype(np.float64, copy=True)
        quantized = np.asarray([
            quantized_score(value) for value in original_scores
        ], dtype=np.float64)
        candidate = int(max(
            positives.tolist(),
            key=lambda index: (float(original_scores[index]), -index)))
        maximum = float(quantized.max())
        top_indices = np.flatnonzero(quantized == maximum)
        if len(top_indices) == 1 and int(top_indices[0]) == candidate:
            summary['image_already_unique_top_tp'] += 1
            continue
        new_score = float(Decimal(str(maximum)) + Decimal('0.01'))
        if quantized_score(new_score) <= maximum:
            raise RuntimeError('promotion did not survive two-decimal scoring')
        rows[candidate, 13] = new_score
        changed_rows[image_id] = {
            'row_index': candidate,
            'old_score': float(original_scores[candidate]),
            'old_quantized_score': float(quantized[candidate]),
            'old_image_quantized_max': maximum,
            'new_score': new_score,
            'new_quantized_score': quantized_score(new_score),
            'assigned_label': int(row_labels[candidate]),
        }
        summary['image_promoted'] += 1
    if (summary['image_without_unique_tp_candidate']
            + summary['image_with_unique_tp_candidate']
            != summary['image_count']):
        raise RuntimeError('promotion image census is inconsistent')
    if summary['image_promoted'] != len(changed_rows):
        raise RuntimeError('promotion row census is inconsistent')
    return promoted, changed_rows, summary


def assert_one_score_change_per_image(original: dict, changed: dict,
                                      expected_changes: dict) -> None:
    observed_changed_images = set()
    for image_id in original:
        first = np.asarray(original[image_id])
        second = np.asarray(changed[image_id])
        if first.shape != second.shape:
            raise RuntimeError('promotion changed prediction shape')
        if not np.array_equal(first[:, :13], second[:, :13]):
            raise RuntimeError('promotion changed class or geometry')
        score_difference = np.flatnonzero(first[:, 13] != second[:, 13])
        if len(score_difference) > 1:
            raise RuntimeError('promotion changed more than one score in an image')
        if len(score_difference) == 1:
            observed_changed_images.add(image_id)
            expected = int(expected_changes[image_id]['row_index'])
            if int(score_difference[0]) != expected:
                raise RuntimeError('promotion changed the wrong row')
    if observed_changed_images != set(expected_changes):
        raise RuntimeError('promotion changed-image set is inconsistent')


def receipt_summary(receipt: dict) -> dict:
    triplet = base.ap_triplet(receipt)
    return {
        'easy': triplet[0],
        'moderate': triplet[1],
        'hard': triplet[2],
    }


def run(args) -> dict:
    started = time.time()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('exp47_top1_tp_promotion')
    runtime = base.runtime_snapshot()
    cache = base.load_cache(args.max_images)
    dataset = base.load_dataset()
    image_ids = [int(value) for value in cache['image_ids'].tolist()]
    selected_dataset = base.subset_dataset(dataset, image_ids)
    formal = args.max_images == 0

    manifest = {
        'started_at': datetime.now(timezone.utc).astimezone().isoformat(),
        'command': [sys.executable, *sys.argv],
        'working_directory': str(Path.cwd()),
        'formal': formal,
        'runtime': runtime,
        'inputs': {
            'cache': str(base.CACHE),
            'cache_sha256': sha256(base.CACHE),
            'checkpoint': str(base.CHECKPOINT),
            'checkpoint_sha256': sha256(base.CHECKPOINT),
            'config': str(base.CONFIG),
            'config_sha256': sha256(base.CONFIG),
            'attribution_helper': str(Path(base.__file__).resolve()),
            'attribution_helper_sha256': sha256(Path(base.__file__).resolve()),
            'script_sha256': sha256(Path(__file__).resolve()),
        },
        'locked_controls': {
            'checkpoint_epoch': 247,
            'strict_car_only_query_count': 50,
            'top50_candidate_set_fixed': True,
            'tp_definition': (
                'score-independent one-to-one Moderate Car exact 3D IoU '
                'Hungarian assignment with IoU > 0.7'),
            'maximum_changed_scores_per_image': 1,
            'selected_tp': 'highest-original-score unique TP candidate',
            'promotion': (
                'smallest 0.01 score grid point strictly above the image '
                'maximum after repository two-decimal quantization'),
            'nms_threshold': args.nms_threshold,
        },
    }
    atomic_json(output_dir / 'initial_manifest.json', manifest)

    original = base.cache_results(cache)
    original_labels, original_iou, original_label_summary = base.aligned_labels(
        selected_dataset, original, logger)
    promoted_no_nms, no_nms_changes, no_nms_promotion_summary = (
        promote_one_tp_per_image(original, original_labels))
    assert_one_score_change_per_image(
        original, promoted_no_nms, no_nms_changes)

    original_nms, original_keep = base.rank_nms(
        original, original_labels, original_iou,
        args.nms_threshold, use_oracle=False)
    promoted_before_nms, promoted_keep = base.rank_nms(
        promoted_no_nms, original_labels, original_iou,
        args.nms_threshold, use_oracle=False)

    original_nms_labels, _, original_nms_label_summary = base.aligned_labels(
        selected_dataset, original_nms, logger)
    promoted_after_nms, after_nms_changes, after_nms_promotion_summary = (
        promote_one_tp_per_image(original_nms, original_nms_labels))
    assert_one_score_change_per_image(
        original_nms, promoted_after_nms, after_nms_changes)

    evaluations = {
        'baseline_no_nms': base.evaluate(
            selected_dataset, original, logger, 'baseline_no_nms'),
        'top1_tp_promoted_no_nms': base.evaluate(
            selected_dataset, promoted_no_nms, logger,
            'top1_tp_promoted_no_nms'),
        'baseline_nms080': base.evaluate(
            selected_dataset, original_nms, logger, 'baseline_nms080'),
        'top1_tp_promoted_before_nms080': base.evaluate(
            selected_dataset, promoted_before_nms, logger,
            'top1_tp_promoted_before_nms080'),
        'top1_tp_promoted_after_fixed_nms080': base.evaluate(
            selected_dataset, promoted_after_nms, logger,
            'top1_tp_promoted_after_fixed_nms080'),
    }
    if formal:
        actual = evaluations['baseline_no_nms']['selection_score']
        if abs(actual - EXPECTED_NO_NMS_MODERATE) > 1e-10:
            raise RuntimeError(
                f'baseline contract failed: {actual} != '
                f'{EXPECTED_NO_NMS_MODERATE}')

    no_nms_baseline = evaluations['baseline_no_nms']['selection_score']
    nms_baseline = evaluations['baseline_nms080']['selection_score']
    summary = {
        'baseline_no_nms_moderate': no_nms_baseline,
        'promoted_no_nms_moderate': evaluations[
            'top1_tp_promoted_no_nms']['selection_score'],
        'delta_no_nms': evaluations[
            'top1_tp_promoted_no_nms']['selection_score'] - no_nms_baseline,
        'baseline_nms080_moderate': nms_baseline,
        'promoted_before_nms080_moderate': evaluations[
            'top1_tp_promoted_before_nms080']['selection_score'],
        'delta_before_nms080': evaluations[
            'top1_tp_promoted_before_nms080']['selection_score'] - nms_baseline,
        'promoted_after_fixed_nms080_moderate': evaluations[
            'top1_tp_promoted_after_fixed_nms080']['selection_score'],
        'delta_after_fixed_nms080': evaluations[
            'top1_tp_promoted_after_fixed_nms080']['selection_score']
            - nms_baseline,
    }
    result = {
        'status': 'passed',
        'diagnostic': 'Exp47 E247 one-TP-per-image minimal score promotion',
        'formal': formal,
        'manifest': manifest,
        'labels': {
            'before_nms': original_label_summary,
            'after_baseline_nms': original_nms_label_summary,
        },
        'promotion': {
            'no_nms_or_before_nms': no_nms_promotion_summary,
            'after_fixed_nms': after_nms_promotion_summary,
        },
        'nms': {
            'baseline_prediction_count': base.prediction_count(original_nms),
            'promoted_before_nms_prediction_count': (
                base.prediction_count(promoted_before_nms)),
            'changed_keep_set_images': int(sum(
                set(original_keep[image_id]) != set(promoted_keep[image_id])
                for image_id in original_keep)),
        },
        'evaluations': evaluations,
        'ap_triplets': {
            name: receipt_summary(receipt)
            for name, receipt in evaluations.items()
        },
        'summary': summary,
        'limitations': [
            'Validation GT is used; the intervention is non-deployable.',
            'TP identity uses a score-independent Hungarian oracle and can '
            'differ from official score-dependent assignment at some thresholds.',
            'Promotion establishes only a per-image top-1 condition; all '
            'remaining TP/FP ordering stays unchanged.',
            'Promoting before NMS can change both NMS survivors and AP order; '
            'the fixed-NMS arm separates the latter effect.',
            'The minimum 0.01 promotion is defined after the repository exact '
            'two-decimal export contract.',
        ],
        'duration_seconds': time.time() - started,
        'finished_at': datetime.now(timezone.utc).astimezone().isoformat(),
    }
    atomic_json(output_dir / 'result.json', result)
    logger.info('FINAL_TOP1_TP_PROMOTION %s', json.dumps(
        summary, ensure_ascii=False, sort_keys=True))
    print('EXP47_TOP1_TP_PROMOTION_OK ' + json.dumps(
        summary, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--nms-threshold', type=float, default=0.8)
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
