#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score
import torch
from torch import nn
from torch.nn import functional as F
import yaml

ROOT = Path('/home/zhangtingyu/Project/Mono3D/MonoDGP')
EXP53_SCRIPT = Path('/tmp/exp47_tp07_frozen_probe.py')
EXP53_OUTPUT = ROOT / 'outputs/V2-0053_实验53_Exp47冻结TP07可靠性探针'
CONFIG = ROOT / 'outputs/V2-0047_实验47_去除三维IoU质量头/resolved_config.yaml'
CHECKPOINT = ROOT / 'outputs/V2-0047_实验47_去除三维IoU质量头/checkpoint_best.pth'
EXPECTED_BASELINE = 24.252757700242608
SEEDS = (54001, 54002, 54003)
LAMBDAS = (0.001, 0.01, 0.1)
IOU_THRESHOLD = 0.7

sys.path.insert(0, str(ROOT))
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.datasets.kitti.kitti_eval_python import kitti_common as kitti
from lib.datasets.kitti.kitti_eval_python.eval import calculate_iou_partly, clean_data


def load_exp53_module():
    if not EXP53_SCRIPT.exists():
        raise FileNotFoundError(f'missing audited cache helper: {EXP53_SCRIPT}')
    spec = importlib.util.spec_from_file_location('exp53_cache_helper', EXP53_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict:
    with CONFIG.open(encoding='utf-8') as handle:
        return yaml.load(handle, Loader=yaml.Loader)


def make_clean_train_dataset(cfg: dict) -> KITTI_Dataset:
    dataset_cfg = copy.deepcopy(cfg['dataset'])
    dataset_cfg['batch_size'] = 1
    dataset = KITTI_Dataset(split='train', cfg=dataset_cfg)
    dataset.data_augmentation = False
    dataset.set_epoch(0)
    return dataset


def cache_clean_train(cfg, output_path: Path, logger, max_images=None):
    helper = load_exp53_module()

    def clean_dataset_factory(_cfg, split):
        if split != 'train':
            raise RuntimeError(f'clean cache only supports train, got {split}')
        return make_clean_train_dataset(_cfg)

    helper.make_dataset = clean_dataset_factory
    model, _ = helper.build_frozen_model(cfg)
    try:
        cache, dataset = helper.cache_split(
            cfg, model, 'train', output_path, logger, max_images=max_images)
    finally:
        del model
        torch.cuda.empty_cache()
    return cache, dataset


def cache_results(cache):
    results = {}
    for index, image_id in enumerate(cache['image_ids'].tolist()):
        rows = cache['detections'][index].numpy().copy()
        rows[:, -1] = cache['base_scores'][index].numpy()
        results[int(image_id)] = rows
    return results


def subset_dataset(dataset: KITTI_Dataset, image_ids) -> KITTI_Dataset:
    selected = copy.copy(dataset)
    selected.idx_list = [str(int(value)) for value in image_ids]
    return selected


def official_aligned_labels(dataset, cache, logger):
    """Create Moderate Car positive/negative/ignore labels.

    Valid Moderate Cars participate in score-independent one-to-one maximum-IoU
    assignment.  A matched query is positive only at IoU > 0.7.  Queries that
    the official Moderate cleaner ignores, or the one query consumed by an
    ignored Car/Van, are ignored.  Every remaining query is a real negative.
    """
    image_ids = [int(value) for value in cache['image_ids'].tolist()]
    local_dataset = subset_dataset(dataset, image_ids)
    dt_annos = local_dataset._decoded_predictions_to_annos(cache_results(cache))
    gt_annos = kitti.get_label_annos(local_dataset.label_dir, image_ids)
    overlaps, _, _, _ = calculate_iou_partly(
        gt_annos, dt_annos, metric=2, num_parts=min(50, len(image_ids)))
    labels = torch.zeros((len(image_ids), 50), dtype=torch.int8)
    assigned_iou = torch.zeros((len(image_ids), 50), dtype=torch.float32)
    positive_count = ignored_count = 0
    for image_index, (gt, dt, overlap) in enumerate(
            zip(gt_annos, dt_annos, overlaps)):
        _, ignored_gt, ignored_dt, _ = clean_data(gt, dt, 0, 1)
        ignored_gt = np.asarray(ignored_gt)
        ignored_dt = np.asarray(ignored_dt)
        valid_gt = np.flatnonzero(ignored_gt == 0)
        ignored_gt_indices = np.flatnonzero(ignored_gt == 1)
        valid_det = np.flatnonzero(ignored_dt == 0)
        query_labels = np.zeros(50, dtype=np.int8)
        query_labels[ignored_dt == 1] = -1
        occupied = set()
        if len(valid_gt) and len(valid_det):
            source, target = linear_sum_assignment(
                -overlap[np.ix_(valid_gt, valid_det)])
            for source_index, target_index in zip(source, target):
                gt_index = valid_gt[source_index]
                det_index = valid_det[target_index]
                value = float(overlap[gt_index, det_index])
                assigned_iou[image_index, det_index] = value
                if value > IOU_THRESHOLD:
                    query_labels[det_index] = 1
                    occupied.add(int(det_index))
        remaining_det = np.asarray(
            [value for value in valid_det if int(value) not in occupied],
            dtype=np.int64)
        if len(ignored_gt_indices) and len(remaining_det):
            source, target = linear_sum_assignment(
                -overlap[np.ix_(ignored_gt_indices, remaining_det)])
            for source_index, target_index in zip(source, target):
                gt_index = ignored_gt_indices[source_index]
                det_index = remaining_det[target_index]
                if float(overlap[gt_index, det_index]) > IOU_THRESHOLD:
                    query_labels[det_index] = -1
        labels[image_index] = torch.from_numpy(query_labels)
        positive_count += int((query_labels == 1).sum())
        ignored_count += int((query_labels == -1).sum())
    logger.info(
        'LABELS images=%d positive=%d negative=%d ignore=%d',
        len(image_ids), positive_count, int((labels == 0).sum()), ignored_count)
    if positive_count == 0:
        raise RuntimeError('official-aligned labels contain no positives')
    return labels, assigned_iou


class ResidualProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, inputs):
        return self.network(inputs).squeeze(-1)


def internal_image_split(image_ids):
    calibration = ((image_ids * 2654435761) % 10) == 0
    if not calibration.any() or calibration.all():
        raise RuntimeError('internal image split is degenerate')
    return ~calibration, calibration


def calibration_ap(model, normalized, base_log, labels, query_mask):
    model.eval()
    valid = query_mask & (labels != -1)
    parts = []
    with torch.no_grad():
        indices = valid.nonzero(as_tuple=False).flatten()
        for begin in range(0, len(indices), 8192):
            batch = indices[begin:begin + 8192]
            parts.append((base_log[batch].cuda()
                          + model(normalized[batch].cuda())).cpu())
    values = torch.cat(parts).numpy()
    targets = labels[valid].numpy()
    return float(average_precision_score(targets, values))


def hard_pairs(model, normalized, base_log, labels, query_mask):
    valid = query_mask & (labels != -1)
    positives = (query_mask & (labels == 1)).nonzero(as_tuple=False).flatten()
    negatives = (query_mask & (labels == 0)).nonzero(as_tuple=False).flatten()
    if len(positives) == 0 or len(negatives) < 4:
        raise RuntimeError('pair sampler requires positives and negatives')
    model.eval()
    score_parts = []
    valid_indices = valid.nonzero(as_tuple=False).flatten()
    with torch.no_grad():
        for begin in range(0, len(valid_indices), 8192):
            batch = valid_indices[begin:begin + 8192]
            score_parts.append((base_log[batch].cuda()
                                + model(normalized[batch].cuda())).cpu())
    scores = torch.empty_like(base_log)
    scores[valid_indices] = torch.cat(score_parts)
    negative_scores, order = torch.sort(scores[negatives])
    sorted_negatives = negatives[order]
    positions = torch.searchsorted(negative_scores, scores[positives])
    offsets = torch.tensor([-1, 0, 1, 2])
    selections = (positions[:, None] + offsets[None, :]).clamp(
        0, len(sorted_negatives) - 1)
    positive_pairs = positives[:, None].expand(-1, 4).reshape(-1)
    negative_pairs = sorted_negatives[selections].reshape(-1)
    return positive_pairs, negative_pairs


def train_candidate(cache, labels_2d, seed, anchor_lambda, logger,
                    max_epochs=50):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    features = cache['features'].reshape(-1, cache['features'].shape[-1]).float()
    labels = labels_2d.reshape(-1).to(torch.int8)
    base_log = cache['base_scores'].reshape(-1).clamp_min(1e-12).log()
    image_train, image_calibration = internal_image_split(cache['image_ids'])
    query_train = image_train[:, None].expand(-1, 50).reshape(-1)
    query_calibration = image_calibration[:, None].expand(-1, 50).reshape(-1)
    feature_train = query_train & (labels != -1)
    mean = features[feature_train].mean(0)
    std = features[feature_train].std(0).clamp_min(1e-6)
    normalized = ((features - mean) / std).float()
    model = ResidualProbe(normalized.shape[1]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_epoch = 0
    best_ap = -1.0
    stale = 0
    history = []
    generator = torch.Generator().manual_seed(seed + int(anchor_lambda * 1e6))
    for epoch in range(1, max_epochs + 1):
        positive_pairs, negative_pairs = hard_pairs(
            model, normalized, base_log, labels, query_train)
        permutation = torch.randperm(len(positive_pairs), generator=generator)
        positive_pairs = positive_pairs[permutation]
        negative_pairs = negative_pairs[permutation]
        model.train()
        loss_sum = rank_sum = anchor_sum = 0.0
        sample_count = 0
        for begin in range(0, len(positive_pairs), 4096):
            pos = positive_pairs[begin:begin + 4096]
            neg = negative_pairs[begin:begin + 4096]
            # The anchor is a global identity-preservation constraint, not a
            # second weight on the selected hard pairs.  Uniformly sample all
            # training Queries, including ignored Queries, so every score is
            # discouraged from moving without ranking evidence.
            train_indices = query_train.nonzero(as_tuple=False).flatten()
            anchor = torch.randint(
                0, len(train_indices), (len(pos),), generator=generator)
            anchor = train_indices[anchor]
            pos_delta = model(normalized[pos].cuda())
            neg_delta = model(normalized[neg].cuda())
            anchor_delta = model(normalized[anchor].cuda())
            pos_score = base_log[pos].cuda() + pos_delta
            neg_score = base_log[neg].cuda() + neg_delta
            rank_loss = F.softplus(-(pos_score - neg_score)).mean()
            anchor_loss = anchor_delta.square().mean()
            loss = rank_loss + float(anchor_lambda) * anchor_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(pos)
            loss_sum += float(loss.detach()) * count
            rank_sum += float(rank_loss.detach()) * count
            anchor_sum += float(anchor_loss.detach()) * count
            sample_count += count
        cal_ap = calibration_ap(
            model, normalized, base_log, labels, query_calibration)
        history.append({
            'epoch': epoch, 'loss': loss_sum / sample_count,
            'rank_loss': rank_sum / sample_count,
            'anchor_loss': anchor_sum / sample_count,
            'calibration_ap': cal_ap,
        })
        if cal_ap > best_ap + 1e-8:
            best_ap = cal_ap
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            logger.info(
                'TRAIN seed=%d lambda=%.4g epoch=%d cal_ap=%.8f best=%.8f',
                seed, anchor_lambda, epoch, cal_ap, best_ap)
        if stale >= 8:
            break
    model.load_state_dict(best_state)
    return model.cpu(), mean, std, {
        'seed': seed, 'anchor_lambda': anchor_lambda,
        'best_epoch': best_epoch, 'best_calibration_ap': best_ap,
        'history': history,
        'train_positive_count': int((query_train & (labels == 1)).sum()),
        'calibration_positive_count': int(
            (query_calibration & (labels == 1)).sum()),
    }


def score_model(model, mean, std, cache):
    features = cache['features'].reshape(-1, cache['features'].shape[-1]).float()
    normalized = ((features - mean) / std).float()
    model = model.cuda().eval()
    deltas = []
    with torch.no_grad():
        for begin in range(0, len(normalized), 8192):
            deltas.append(model(normalized[begin:begin + 8192].cuda()).cpu())
    delta = torch.cat(deltas).reshape(cache['base_scores'].shape)
    score = cache['base_scores'] * torch.exp(delta)
    if not torch.isfinite(score).all():
        raise RuntimeError('unbounded residual emitted a non-finite score')
    return score, delta


def evaluate(dataset, cache, scores, logger, name):
    logger.info('EVALUATING_%s', name)
    results = {}
    for index, image_id in enumerate(cache['image_ids'].tolist()):
        rows = cache['detections'][index].numpy().copy()
        rows[:, -1] = scores[index].numpy().astype(np.float32, copy=False)
        results[int(image_id)] = rows
    return dataset.eval(results, logger, return_metrics=True)


def run(output_dir: Path, smoke: bool, logger, reuse_clean_cache=None):
    helper = load_exp53_module()
    runtime = helper.configure_runtime()
    initial_manifest = {
        'started_at': datetime.now(timezone.utc).astimezone().isoformat(),
        'argv': sys.argv,
        'working_directory': str(Path.cwd()),
        'runtime': runtime,
        'checkpoint': str(CHECKPOINT),
        'checkpoint_sha256': sha256(CHECKPOINT),
        'exp53_helper': str(EXP53_SCRIPT),
        'exp53_helper_sha256': sha256(EXP53_SCRIPT),
        'formal': not smoke,
    }
    (output_dir / 'initial_manifest.json').write_text(
        json.dumps(initial_manifest, ensure_ascii=False, indent=2),
        encoding='utf-8')
    (output_dir / 'command.txt').write_text(
        ' '.join(sys.argv) + '\n', encoding='utf-8')
    cfg = load_config()
    if smoke:
        val_cache = torch.load(
            EXP53_OUTPUT / 'val_cache.pt', map_location='cpu', weights_only=False)
        count = 256
        cache = {key: (value[:count] if torch.is_tensor(value)
                       and value.shape[:1] == val_cache['features'].shape[:1]
                       else value)
                 for key, value in val_cache.items()}
        dataset_cfg = copy.deepcopy(cfg['dataset'])
        dataset_cfg['batch_size'] = 1
        dataset = KITTI_Dataset(split='val', cfg=dataset_cfg)
        labels, _ = official_aligned_labels(dataset, cache, logger)
        probe, mean, std, training = train_candidate(
            cache, labels, SEEDS[0], LAMBDAS[1], logger, max_epochs=3)
        scores, delta = score_model(probe, mean, std, cache)
        result = {
            'status': 'passed', 'images': count,
            'positive_count': int((labels == 1).sum()),
            'ignore_count': int((labels == -1).sum()),
            'delta_abs_max': float(delta.abs().max()),
            'training': training,
        }
        (output_dir / 'smoke_result.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print('EXP54_SMOKE_OK ' + json.dumps(result, ensure_ascii=False), flush=True)
        return

    if reuse_clean_cache is None:
        clean_cache_path = output_dir / 'clean_train_cache.pt'
        train_cache, train_dataset = cache_clean_train(
            cfg, clean_cache_path, logger)
    else:
        clean_cache_path = Path(reuse_clean_cache).resolve()
        train_cache = torch.load(
            clean_cache_path, map_location='cpu', weights_only=False)
        train_dataset = make_clean_train_dataset(cfg)
        logger.info('REUSED_CLEAN_CACHE=%s', clean_cache_path)
    val_cache = torch.load(
        EXP53_OUTPUT / 'val_cache.pt', map_location='cpu', weights_only=False)
    dataset_cfg = copy.deepcopy(cfg['dataset'])
    dataset_cfg['batch_size'] = 1
    val_dataset = KITTI_Dataset(split='val', cfg=dataset_cfg)
    train_labels, train_iou = official_aligned_labels(
        train_dataset, train_cache, logger)
    val_labels, val_iou = official_aligned_labels(
        val_dataset, val_cache, logger)
    torch.save({'labels': train_labels, 'assigned_iou': train_iou},
               output_dir / 'clean_train_labels.pt')
    torch.save({'labels': val_labels, 'assigned_iou': val_iou},
               output_dir / 'val_labels.pt')
    baseline = evaluate(
        val_dataset, val_cache, val_cache['base_scores'], logger, 'baseline')
    if abs(float(baseline['selection_score']) - EXPECTED_BASELINE) > 1e-10:
        raise RuntimeError(
            f"baseline contract failed: {baseline['selection_score']}")

    seed_reports = []
    for seed in SEEDS:
        candidates = []
        for anchor_lambda in LAMBDAS:
            probe, mean, std, training = train_candidate(
                train_cache, train_labels, seed, anchor_lambda, logger)
            candidates.append((training['best_calibration_ap'], anchor_lambda,
                               probe, mean, std, training))
        selected = max(candidates, key=lambda item: (item[0], -item[1]))
        _, anchor_lambda, probe, mean, std, training = selected
        scores, delta = score_model(probe, mean, std, val_cache)
        evaluation = evaluate(
            val_dataset, val_cache, scores, logger, f'seed_{seed}')
        seed_dir = output_dir / f'seed_{seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': probe.state_dict(), 'mean': mean, 'std': std,
            'training': training,
        }, seed_dir / 'probe.pt')
        report = {
            'seed': seed, 'selected_anchor_lambda': anchor_lambda,
            'candidate_calibration_ap': {
                str(item[1]): item[0] for item in candidates},
            'training': training,
            'delta': {
                'mean': float(delta.mean()), 'std': float(delta.std()),
                'abs_max': float(delta.abs().max()),
            },
            'evaluation': evaluation,
        }
        (seed_dir / 'result.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        seed_reports.append(report)

    moderate = np.asarray([
        item['evaluation']['selection_score'] for item in seed_reports])
    result = {
        'status': 'passed',
        'experiment': 'Exp54 frozen baseline-anchored hard-pair rank probe',
        'method': {
            'detector_frozen': True, 'alpha': None, 'sigmoid': False,
            'score': 'c*d*exp(delta)',
            'loss': 'hard-pair softplus rank loss + lambda*mean(delta^2)',
            'negative_sampling': 'four nearest-score negatives per positive, global across images',
            'label': 'Moderate Car positive/negative/ignore with exact official 3D overlap',
            'lambda_candidates': LAMBDAS,
            'lambda_selection': 'training internal image-level calibration AP only',
            'validation_used_for_training_or_selection': False,
        },
        'runtime': runtime,
        'checkpoint': {'path': str(CHECKPOINT), 'sha256': sha256(CHECKPOINT)},
        'artifacts': {
            'clean_train_cache_sha256': sha256(clean_cache_path),
            'source_helper_sha256': sha256(EXP53_SCRIPT),
        },
        'labels': {
            'train_positive': int((train_labels == 1).sum()),
            'train_negative': int((train_labels == 0).sum()),
            'train_ignore': int((train_labels == -1).sum()),
            'val_positive': int((val_labels == 1).sum()),
            'val_negative': int((val_labels == 0).sum()),
            'val_ignore': int((val_labels == -1).sum()),
        },
        'baseline': baseline,
        'seeds': seed_reports,
        'summary': {
            'moderate_scores': moderate.tolist(),
            'moderate_mean': float(moderate.mean()),
            'moderate_std_population': float(moderate.std()),
            'mean_delta_vs_baseline': float(
                moderate.mean() - float(baseline['selection_score'])),
        },
    }
    (output_dir / 'result.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('EXP54_RESULT ' + json.dumps(result['summary'], ensure_ascii=False),
          flush=True)
    print('EXP54_OK', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--reuse-clean-cache')
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(output_dir / 'run.log'),
                  logging.StreamHandler(sys.stdout)])
    run(output_dir, args.smoke, logging.getLogger('exp54'),
        reuse_clean_cache=args.reuse_clean_cache)


if __name__ == '__main__':
    main()
