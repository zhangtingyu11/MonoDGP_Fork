"""All-query 3D-IoU quality regression and within-GT ranking loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def all_query_quality_ranking_loss(
        prediction: Tensor,
        iou3d_matrix: Tensor,
        target_sizes: tuple[int, ...],
        group_num: int,
        ranking_iou_gap: float = 0.1,
        low_iou_threshold: float = 0.1,
        low_iou_weight: float = 0.1,
        full_weight_iou: float = 0.5) -> dict[str, Tensor]:
    """Supervise every query and rank candidates assigned to the same GT.

    Each query is assigned to the GT with its maximum exact 3D IoU. Pointwise
    supervision uses that maximum IoU. Pairwise supervision is constructed
    only inside one decoder query group and one assigned GT, and only when the
    two IoUs differ by at least ``ranking_iou_gap``.

    The IoU matrix is treated as a numerical target. Gradients flow through
    ``prediction`` into the quality head and shared decoder features, never
    through the target construction.
    """
    if prediction.ndim == 3 and prediction.shape[-1] == 1:
        prediction = prediction[..., 0]
    if prediction.ndim != 2:
        raise ValueError('quality prediction must have shape [B,Q] or [B,Q,1]')
    if iou3d_matrix.ndim != 3:
        raise ValueError('3D-IoU matrix must have shape [B,Q,T]')
    if iou3d_matrix.shape[:2] != prediction.shape:
        raise ValueError('3D-IoU matrix does not match quality prediction')
    if len(target_sizes) != prediction.shape[0]:
        raise ValueError('target_sizes does not match batch size')
    if group_num <= 0 or prediction.shape[1] % group_num:
        raise ValueError('query count must be divisible by group_num')
    if ranking_iou_gap <= 0 or ranking_iou_gap > 1:
        raise ValueError('ranking_iou_gap must be in (0, 1]')
    if low_iou_threshold < 0 or low_iou_threshold > 1:
        raise ValueError('low_iou_threshold must be in [0, 1]')
    if low_iou_weight < 0 or low_iou_weight > 1:
        raise ValueError('low_iou_weight must be in [0, 1]')
    if full_weight_iou <= low_iou_threshold or full_weight_iou > 1:
        raise ValueError(
            'full_weight_iou must satisfy low_iou_threshold < value <= 1')

    device = prediction.device
    detached_iou = iou3d_matrix.detach().clamp(0, 1)
    max_iou = prediction.new_zeros(prediction.shape)
    assigned_gt = torch.full(
        prediction.shape, -1, dtype=torch.long, device=device)
    for batch_index, target_count in enumerate(target_sizes):
        target_count = int(target_count)
        if target_count < 0 or target_count > detached_iou.shape[2]:
            raise ValueError('invalid target count for padded IoU matrix')
        if target_count:
            values, indices = detached_iou[
                batch_index, :, :target_count].max(dim=-1)
            max_iou[batch_index] = values
            assigned_gt[batch_index] = indices

    encoded_target = 2.0 * (max_iou - 0.5)
    transition = ((max_iou - low_iou_threshold)
                  / (full_weight_iou - low_iou_threshold)).clamp(0, 1)
    point_weights = low_iou_weight + (1.0 - low_iou_weight) * transition
    point_values = F.smooth_l1_loss(
        prediction, encoded_target, reduction='none')
    # Keep ``low_iou_weight`` as an absolute per-query multiplier. Dividing
    # by the sum of weights would cancel the requested 0.1 attenuation on a
    # batch dominated by near-zero-IoU queries and make the objective scale
    # depend on the positive/negative mixture.
    point_denominator = max(point_values.numel(), 1)
    point_loss = (point_values * point_weights).sum() / point_denominator

    queries_per_group = prediction.shape[1] // group_num
    upper_triangle = torch.triu(
        torch.ones(
            (queries_per_group, queries_per_group),
            dtype=torch.bool, device=device), diagonal=1)
    rank_loss_sum = prediction.new_zeros(())
    rank_correct = prediction.new_zeros(())
    rank_pair_count = prediction.new_zeros(())
    for batch_index, target_count in enumerate(target_sizes):
        if not int(target_count):
            continue
        for group_index in range(group_num):
            begin = group_index * queries_per_group
            end = begin + queries_per_group
            group_prediction = prediction[batch_index, begin:end]
            group_iou = max_iou[batch_index, begin:end]
            group_assignment = assigned_gt[batch_index, begin:end]
            iou_difference = group_iou[:, None] - group_iou[None, :]
            same_gt = (
                (group_assignment[:, None] == group_assignment[None, :])
                & (group_assignment[:, None] >= 0))
            valid = (
                upper_triangle
                & same_gt
                & (iou_difference.abs() >= ranking_iou_gap))
            if not valid.any():
                continue
            prediction_difference = (
                group_prediction[:, None] - group_prediction[None, :])
            direction = torch.sign(iou_difference[valid])
            signed_margin = direction * prediction_difference[valid]
            rank_loss_sum = rank_loss_sum + F.softplus(
                -signed_margin).sum()
            rank_correct = rank_correct + (signed_margin > 0).sum()
            rank_pair_count = rank_pair_count + valid.sum()

    rank_loss = torch.where(
        rank_pair_count > 0,
        rank_loss_sum / rank_pair_count.clamp_min(1.0),
        prediction.sum() * 0.0)
    decoded_prediction = ((prediction + 1.0) * 0.5).clamp(0, 1)
    query_count = max_iou.numel()
    query_denominator = max(query_count, 1)
    high_quality = max_iou >= 0.7
    medium_quality = (max_iou >= 0.5) & (max_iou < 0.7)
    low_nonzero = (
        (max_iou >= low_iou_threshold) & (max_iou < 0.5))
    near_zero = max_iou < low_iou_threshold

    return {
        'loss_quality_point': point_loss,
        'loss_quality_rank': rank_loss,
        'monitor_quality_iou_mae': (
            decoded_prediction.detach() - max_iou).abs().mean(),
        'monitor_quality_target_iou_mean': max_iou.mean(),
        'monitor_quality_predicted_iou_mean': (
            decoded_prediction.detach().mean()),
        'monitor_quality_rank_pair_count': rank_pair_count.detach(),
        'monitor_quality_rank_pair_accuracy': torch.where(
            rank_pair_count > 0,
            rank_correct / rank_pair_count.clamp_min(1.0),
            rank_correct),
        'monitor_quality_point_effective_weight_mean': point_weights.mean(),
        'monitor_quality_iou_lt_0_1_fraction': (
            near_zero.sum() / query_denominator),
        'monitor_quality_iou_0_1_to_0_5_fraction': (
            low_nonzero.sum() / query_denominator),
        'monitor_quality_iou_0_5_to_0_7_fraction': (
            medium_quality.sum() / query_denominator),
        'monitor_quality_iou_ge_0_7_fraction': (
            high_quality.sum() / query_denominator),
    }
