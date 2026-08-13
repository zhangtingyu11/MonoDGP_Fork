"""Freeze a checkpoint and sweep 3D-IoU Hungarian matching weights.

This diagnostic never computes gradients or updates model state.  It keeps the
production 2D matching cost exactly and adds ``-weight * pairwise_iou3d``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.model_helper import build_model
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.utils_helper import set_random_seed
from lib.losses.asymmetric_interval_depth_loss import (
    _centers_on_projected_rays,
    _decode_alpha,
    _decode_yaw_like_public_decoder,
    paired_iou3d,
)
from utils.box_ops import box_cxcylrtb_to_xyxy, generalized_box_iou


class _Logger:
    def info(self, message, *args):
        if args:
            message = message % args
        print(message, flush=True)


def _prepare_targets(raw_targets, device, batch_size):
    moved = {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in raw_targets.items()
    }
    mask = moved["mask_2d"]
    fields = {
        "labels", "boxes", "calibs", "depth", "size_3d", "heading_bin",
        "heading_res", "boxes_3d", "src_size_3d", "depth_unit_scale",
        "projective_rotation_y",
    }
    per_image = []
    for batch_index in range(batch_size):
        item = {}
        for key, value in moved.items():
            if key in fields:
                item[key] = value[batch_index][mask[batch_index]]
            elif key in (
                    "img_size", "projective_input_size",
                    "projective_image_effective_calib"):
                item[key] = value[batch_index]
        per_image.append(item)
    return per_image


def _pairwise_iou3d(outputs, target, class_mean_sizes):
    boxes = outputs["pred_boxes"]
    query_count = boxes.shape[0]
    target_count = target["labels"].shape[0]
    if target_count == 0:
        return boxes.new_zeros((query_count, 0))

    dtype, device = boxes.dtype, boxes.device
    input_size = target["projective_input_size"].to(device=device, dtype=dtype)
    effective_calib = target["projective_image_effective_calib"].to(
        device=device, dtype=dtype)
    predicted_uv = boxes[:, :2] * input_size
    predicted_depth = outputs["pred_depth"][:, 0]
    predicted_center, predicted_ray_valid = _centers_on_projected_rays(
        predicted_uv, predicted_depth[:, None],
        effective_calib.expand(query_count, -1, -1))

    means = torch.as_tensor(class_mean_sizes, device=device, dtype=dtype)
    # Dimensions are class residuals.  For a candidate query/GT pair, decode
    # using that GT class, just as assignment would supervise that pair.
    predicted_dimensions = (
        outputs["pred_3d_dim"][:, None, :]
        + means[target["labels"].long()][None, :, :]).clamp_min(0.05)

    predicted_alpha = _decode_alpha(outputs["pred_angle"])
    image_width = target["img_size"].to(device=device, dtype=dtype)[0]
    heading_calib = target["calibs"][0].to(device=device, dtype=dtype)
    predicted_yaw = _decode_yaw_like_public_decoder(
        predicted_alpha, boxes,
        image_width.expand(query_count),
        heading_calib.expand(query_count, -1, -1))

    target_uv = target["boxes_3d"][:, :2].to(dtype=dtype) * input_size
    target_depth = (
        target["depth"].reshape(-1).to(dtype=dtype)
        / target["depth_unit_scale"].reshape(-1).to(dtype=dtype).clamp_min(1e-6))
    target_center, target_ray_valid = _centers_on_projected_rays(
        target_uv, target_depth[:, None],
        effective_calib.expand(target_count, -1, -1))
    target_dimensions = target["src_size_3d"].to(dtype=dtype)
    target_yaw = target["projective_rotation_y"].reshape(-1).to(dtype=dtype)

    q_index = torch.arange(query_count, device=device).repeat_interleave(target_count)
    t_index = torch.arange(target_count, device=device).repeat(query_count)
    iou = paired_iou3d(
        predicted_center[q_index],
        predicted_dimensions[q_index, t_index],
        predicted_yaw[q_index],
        target_center[t_index],
        target_dimensions[t_index],
        target_yaw[t_index]).reshape(query_count, target_count)
    valid = predicted_ray_valid[:, None] & target_ray_valid[None, :]
    return torch.where(valid, iou, torch.zeros_like(iou)).clamp(0, 1)


def _cost_matrices(outputs, target, class_mean_sizes, matcher):
    labels = target["labels"].long()
    probabilities = outputs["pred_logits"].sigmoid()
    alpha, gamma = 0.25, 2.0
    negative = ((1 - alpha) * probabilities.pow(gamma)
                * (-(1 - probabilities + 1e-8).log()))
    positive = (alpha * (1 - probabilities).pow(gamma)
                * (-(probabilities + 1e-8).log()))
    cost_class = (positive - negative)[:, labels]
    target_boxes = target["boxes_3d"]
    cost_center = torch.cdist(outputs["pred_boxes"][:, :2], target_boxes[:, :2], p=1)
    cost_bbox = torch.cdist(outputs["pred_boxes"][:, 2:6], target_boxes[:, 2:6], p=1)
    prediction_xyxy = box_cxcylrtb_to_xyxy(outputs["pred_boxes"])
    target_xyxy = box_cxcylrtb_to_xyxy(target_boxes)
    giou = generalized_box_iou(prediction_xyxy, target_xyxy)
    cost_2d = (matcher.cost_class * cost_class
               + matcher.cost_3dcenter * cost_center
               + matcher.cost_bbox * cost_bbox
               - matcher.cost_giou * giou)
    iou3d = _pairwise_iou3d(outputs, target, class_mean_sizes)
    target_scores = probabilities[:, labels]
    return {
        "cost_2d": cost_2d,
        "iou3d": iou3d,
        "giou2d": giou,
        "center_l1": cost_center,
        "bbox_l1": cost_bbox,
        "class_score": target_scores,
    }


def _append_selected(storage, matrices, sources, targets):
    for name, matrix in matrices.items():
        storage[name].extend(matrix[sources, targets].detach().cpu().double().tolist())


def _summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--weights", type=float, nargs="+",
        default=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.load(handle, Loader=yaml.Loader)
    cfg["dataset"]["batch_size"] = args.batch_size
    set_random_seed(cfg.get("random_seed", 444))
    _, loader = build_dataloader(cfg["dataset"], workers=args.workers)
    model, criterion = build_model(cfg["model"])
    device = torch.device("cuda")
    model.to(device).eval()
    criterion.to(device).eval()
    epoch, best_result, best_epoch = load_checkpoint(
        model, None, args.checkpoint, device, logger=_Logger())

    weights = tuple(float(weight) for weight in args.weights)
    selected = {weight: defaultdict(list) for weight in weights}
    changed = {weight: 0 for weight in weights}
    improved = {weight: defaultdict(int) for weight in weights}
    total_targets = 0
    image_count = 0
    start = time.time()
    means = loader.dataset.cls_mean_size
    matcher = criterion.matcher

    with torch.inference_mode():
        for batch_index, (inputs, calibs, raw_targets, info) in enumerate(loader):
            if args.max_images is not None and image_count >= args.max_images:
                break
            inputs = inputs.to(device, non_blocking=True)
            calibs = calibs.to(device, non_blocking=True)
            img_sizes = info["img_size"].to(device, non_blocking=True)
            targets = _prepare_targets(raw_targets, device, inputs.shape[0])
            outputs = model(inputs, calibs, raw_targets, img_sizes, dn_args=0)
            for image_index, target in enumerate(targets):
                target_count = len(target["labels"])
                if target_count == 0:
                    continue
                per_image_outputs = {
                    key: outputs[key][image_index]
                    for key in (
                        "pred_logits", "pred_boxes", "pred_depth",
                        "pred_3d_dim", "pred_angle")}
                matrices = _cost_matrices(
                    per_image_outputs, target, means, matcher)
                numpy_2d = matrices["cost_2d"].detach().cpu().double().numpy()
                numpy_iou3d = matrices["iou3d"].detach().cpu().double().numpy()
                assignments = {}
                for weight in weights:
                    source, target_index = linear_sum_assignment(
                        numpy_2d - weight * numpy_iou3d)
                    order = np.argsort(target_index)
                    assignments[weight] = (
                        source[order], target_index[order])
                    _append_selected(
                        selected[weight], matrices,
                        source[order], target_index[order])

                baseline_source, baseline_target = assignments[0.0]
                baseline_iou = numpy_iou3d[baseline_source, baseline_target]
                for weight in weights:
                    source, target_index = assignments[weight]
                    candidate_iou = numpy_iou3d[source, target_index]
                    delta = candidate_iou - baseline_iou
                    changed[weight] += int(np.count_nonzero(source != baseline_source))
                    improved[weight]["iou_gain_gt_0"] += int(np.count_nonzero(delta > 1e-9))
                    improved[weight]["iou_gain_ge_0_05"] += int(np.count_nonzero(delta >= 0.05))
                    improved[weight]["iou_gain_ge_0_10"] += int(np.count_nonzero(delta >= 0.10))
                    improved[weight]["iou_loss_gt_0"] += int(np.count_nonzero(delta < -1e-9))
                    selected[weight]["iou3d_delta"].extend(delta.tolist())
                total_targets += target_count
            image_count += inputs.shape[0]
            if (batch_index + 1) % 250 == 0:
                print(
                    f"processed {image_count}/{len(loader.dataset)} images",
                    flush=True)

    results = {
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": int(epoch),
        "checkpoint_best_result": float(best_result),
        "checkpoint_best_epoch": int(best_epoch),
        "images": image_count,
        "matched_targets": total_targets,
        "batch_size": args.batch_size,
        "seconds": time.time() - start,
        "formula": "cost_2d - lambda_3d * iou3d",
        "weights": {},
    }
    for weight in weights:
        entry = {
            name: _summarize(values)
            for name, values in selected[weight].items()
        }
        denominator = max(total_targets, 1)
        entry.update({
            "changed_count": changed[weight],
            "changed_fraction": changed[weight] / denominator,
            **{
                name: count / denominator
                for name, count in improved[weight].items()
            },
        })
        results["weights"][str(weight)] = entry

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
