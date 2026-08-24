"""NMS-aware pairwise ranking for all-query 3D-IoU classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from lib.losses.asymmetric_interval_depth_loss import (
    _decode_alpha,
    _decode_yaw_like_public_decoder,
    _inside_precomputed,
    _intersection_area_from_corners,
    _rectangle_corners,
)


@torch.no_grad()
def _decode_query_bev_geometry_batched(
        outputs: dict[str, Tensor], targets: list[dict[str, Tensor]],
        assigned_labels: Tensor, decode_mean_sizes: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Decode all images and queries in one GPU batch."""
    boxes = outputs['pred_boxes']
    batch_size, query_count = boxes.shape[:2]
    device, dtype = boxes.device, boxes.dtype
    input_size = torch.stack(tuple(
        target['projective_input_size'].to(device=device, dtype=dtype)
        for target in targets))
    effective_calib = torch.stack(tuple(
        target['projective_image_effective_calib'].to(
            device=device, dtype=dtype)
        for target in targets))

    pred_uv = boxes[..., :2] * input_size[:, None, :]
    pred_depth = outputs['pred_depth'][..., 0]
    u, v = pred_uv.unbind(-1)
    z = pred_depth
    p = effective_calib[:, None, :, :].expand(
        batch_size, query_count, -1, -1)
    p00, p01, p02, p03 = p[..., 0, :].unbind(-1)
    p10, p11, p12, p13 = p[..., 1, :].unbind(-1)
    p20, p21, p22, p23 = p[..., 2, :].unbind(-1)
    denominator_z = p22 * z + p23
    matrix = torch.stack((
        torch.stack((p00 - u * p20, p01 - u * p21), dim=-1),
        torch.stack((p10 - v * p20, p11 - v * p21), dim=-1)), dim=-2)
    right = torch.stack((
        u * denominator_z - p02 * z - p03,
        v * denominator_z - p12 * z - p13), dim=-1)
    determinant = torch.linalg.det(matrix)
    invalid = determinant.abs() <= 1e-9
    identity = torch.eye(2, device=device, dtype=dtype).expand_as(matrix)
    safe_matrix = torch.where(
        invalid[..., None, None], identity, matrix)
    xy = torch.linalg.solve(
        safe_matrix, right.unsqueeze(-1)).squeeze(-1)
    pred_center = torch.cat((xy, z.unsqueeze(-1)), dim=-1)
    ray_valid = ~invalid

    means = decode_mean_sizes.to(device=device, dtype=dtype)
    dimensions = (
        outputs['pred_3d_dim'] + means[assigned_labels]).clamp_min(0.05)
    _, width, length = dimensions.unbind(-1)

    alpha = _decode_alpha(outputs['pred_angle'])
    if all('physical_ray_heading' in target for target in targets):
        yaw = torch.remainder(
            alpha + torch.atan2(pred_center[..., 0], pred_center[..., 2])
            + torch.pi, 2.0 * torch.pi) - torch.pi
    else:
        image_width = torch.stack(tuple(
            target['img_size'].to(device=device, dtype=dtype)[0]
            for target in targets))
        heading_calib = torch.stack(tuple(
            target['calibs'][0].to(device=device, dtype=dtype)
            for target in targets))
        yaw = _decode_yaw_like_public_decoder(
            alpha, boxes, image_width[:, None].expand(-1, query_count),
            heading_calib[:, None, :, :].expand(
                -1, query_count, -1, -1))

    corners = _rectangle_corners(
        pred_center[..., (0, 2)], width, length, yaw)
    areas = width * length
    valid = (
        ray_valid
        & torch.isfinite(pred_center).all(-1)
        & torch.isfinite(dimensions).all(-1)
        & torch.isfinite(yaw))
    return corners, areas, width, length, yaw, valid


@torch.no_grad()
def _triggered_nms_pairs_batched(
        outputs: dict[str, Tensor], targets: list[dict[str, Tensor]],
        assigned_target: Tensor, assigned_labels: Tensor,
        target_counts: Tensor, decode_mean_sizes: Tensor,
        group_num: int, bev_iou_threshold: float
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return all within-image pairs that would mutually trigger BEV NMS."""
    corners, areas, width, length, yaw, geometry_valid = (
        _decode_query_bev_geometry_batched(
            outputs, targets, assigned_labels, decode_mean_sizes))
    batch_size, query_count = corners.shape[:2]
    queries_per_group = query_count // group_num
    grouped_corners = corners.reshape(
        batch_size, group_num, queries_per_group, 4, 2)
    grouped_min = grouped_corners.amin(dim=-2)
    grouped_max = grouped_corners.amax(dim=-2)
    grouped_target = assigned_target.reshape(
        batch_size, group_num, queries_per_group)
    grouped_valid = geometry_valid.reshape(
        batch_size, group_num, queries_per_group)
    upper = torch.triu(torch.ones(
        (queries_per_group, queries_per_group), dtype=torch.bool,
        device=corners.device), diagonal=1)[None, None, :, :]
    valid = (
        upper
        & (grouped_target[..., :, None] == grouped_target[..., None, :])
        & grouped_valid[..., :, None]
        & grouped_valid[..., None, :]
        & (target_counts[:, None, None, None] > 0)
        & (grouped_max[..., :, None, 0] >= grouped_min[..., None, :, 0])
        & (grouped_max[..., None, :, 0] >= grouped_min[..., :, None, 0])
        & (grouped_max[..., :, None, 1] >= grouped_min[..., None, :, 1])
        & (grouped_max[..., None, :, 1] >= grouped_min[..., :, None, 1]))
    batch, group, first_local, second_local = valid.nonzero(
        as_tuple=True)
    first = group * queries_per_group + first_local
    second = group * queries_per_group + second_local
    first_corners = corners[batch, first]
    second_corners = corners[batch, second]
    first_center = first_corners.mean(dim=-2)
    second_center = second_corners.mean(dim=-2)
    first_inside = _inside_precomputed(
        first_corners, second_center,
        width[batch, second] * 0.5, length[batch, second] * 0.5,
        yaw[batch, second].cos(), yaw[batch, second].sin())
    second_inside = _inside_precomputed(
        second_corners, first_center,
        width[batch, first] * 0.5, length[batch, first] * 0.5,
        yaw[batch, first].cos(), yaw[batch, first].sin())
    intersection = _intersection_area_from_corners(
        first_corners, second_corners, first_inside, second_inside)
    union = areas[batch, first] + areas[batch, second] - intersection
    bev_iou = intersection / union.clamp_min(
        torch.finfo(corners.dtype).eps)
    triggered = bev_iou >= bev_iou_threshold
    return (
        batch[triggered], first[triggered], second[triggered],
        bev_iou[triggered])


def nms_aware_iou_ranking_loss(
        src_logits: Tensor, outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]], iou3d_matrix: Tensor,
        decode_mean_sizes: Tensor, group_num: int,
        bev_iou_threshold: float = 0.8,
        min_iou_delta: float = 1e-6) -> dict[str, Tensor]:
    """Rank only same-GT query pairs that can suppress each other at NMS.

    Pair construction, nearest-GT assignment, BEV overlap and true-IoU
    ordering are detached. The RankNet term sends gradients only through the
    classification logits. Each pair is weighted by its absolute true 3D-IoU
    gap, so a barely distinguishable pair contributes less than a pair whose
    incorrect order would discard a substantially better box.
    """
    if src_logits.ndim != 3:
        raise ValueError('classification logits must have shape [B,Q,C]')
    if iou3d_matrix.shape[:2] != src_logits.shape[:2]:
        raise ValueError('3D-IoU matrix does not match classification logits')
    if len(targets) != src_logits.shape[0]:
        raise ValueError('target batch size does not match logits')
    if group_num <= 0 or src_logits.shape[1] % group_num:
        raise ValueError('query count must be divisible by group_num')
    if not 0.0 <= bev_iou_threshold <= 1.0:
        raise ValueError('BEV NMS threshold must be in [0, 1]')
    if min_iou_delta < 0.0 or min_iou_delta >= 1.0:
        raise ValueError('minimum IoU delta must be in [0, 1)')

    loss_sum = src_logits.new_zeros(())
    pair_count = src_logits.new_zeros(())
    correct_count = src_logits.new_zeros(())
    weighted_correct = src_logits.new_zeros(())
    weight_sum = src_logits.new_zeros(())
    iou_gap_sum = src_logits.new_zeros(())
    inverted_gap_sum = src_logits.new_zeros(())
    inverted_count = src_logits.new_zeros(())
    threshold_crossing_count = src_logits.new_zeros(())
    threshold_crossing_wrong = src_logits.new_zeros(())
    bev_iou_sum = src_logits.new_zeros(())

    target_slots = iou3d_matrix.shape[-1]
    if target_slots == 0:
        zero = src_logits.sum() * 0.0
        return {
            'loss_iou_classification_nms_rank': zero,
            'monitor_nms_rank_pair_count': zero.detach(),
            'monitor_nms_rank_pair_accuracy': zero.detach(),
            'monitor_nms_rank_weighted_accuracy': zero.detach(),
            'monitor_nms_rank_inversion_count': zero.detach(),
            'monitor_nms_rank_iou_gap_mean': zero.detach(),
            'monitor_nms_rank_inverted_iou_gap_mean': zero.detach(),
            'monitor_nms_rank_bev_iou_mean': zero.detach(),
            'monitor_nms_rank_cross_0_7_pair_count': zero.detach(),
            'monitor_nms_rank_cross_0_7_wrong_fraction': zero.detach(),
        }
    device = src_logits.device
    target_counts = torch.as_tensor(
        [target['labels'].numel() for target in targets],
        device=device, dtype=torch.long)
    slot = torch.arange(target_slots, device=device)
    valid_target = slot[None, :] < target_counts[:, None]
    detached_iou = iou3d_matrix.detach().clamp(0, 1)
    masked_iou = torch.where(
        valid_target[:, None, :], detached_iou,
        torch.full_like(detached_iou, -1.0))
    max_iou, assigned_target = masked_iou.max(dim=-1)
    max_iou = torch.where(
        target_counts[:, None] > 0, max_iou, torch.zeros_like(max_iou))
    padded_labels = torch.zeros(
        len(targets), target_slots, device=device, dtype=torch.long)
    for batch_index, target in enumerate(targets):
        count = int(target['labels'].numel())
        padded_labels[batch_index, :count] = target['labels'].reshape(
            -1).to(device=device, dtype=torch.long)
    assigned_labels = padded_labels.gather(1, assigned_target)
    batch, first, second, pair_bev_iou = _triggered_nms_pairs_batched(
        outputs, targets, assigned_target, assigned_labels, target_counts,
        decode_mean_sizes=decode_mean_sizes, group_num=group_num,
        bev_iou_threshold=bev_iou_threshold)
    iou_delta = max_iou[batch, first] - max_iou[batch, second]
    keep = iou_delta.abs() > min_iou_delta
    batch, first, second = batch[keep], first[keep], second[keep]
    pair_bev_iou = pair_bev_iou[keep]
    iou_delta = iou_delta[keep]
    logits = src_logits.gather(
        2, assigned_labels.unsqueeze(-1)).squeeze(-1)
    signed_margin = iou_delta.sign() * (
        logits[batch, first] - logits[batch, second])
    weights = iou_delta.abs()
    pair_loss = weights * F.softplus(-signed_margin)
    correct = signed_margin > 0
    inverted = ~correct
    threshold_crossing = (
        ((max_iou[batch, first] >= 0.7)
         & (max_iou[batch, second] < 0.7))
        | ((max_iou[batch, second] >= 0.7)
           & (max_iou[batch, first] < 0.7)))

    loss_sum = pair_loss.sum()
    pair_count = src_logits.new_tensor(float(first.numel()))
    correct_count = correct.sum()
    weighted_correct = weights[correct].sum()
    weight_sum = weights.sum()
    iou_gap_sum = weights.sum()
    inverted_gap_sum = weights[inverted].sum()
    inverted_count = inverted.sum()
    threshold_crossing_count = threshold_crossing.sum()
    threshold_crossing_wrong = (threshold_crossing & inverted).sum()
    bev_iou_sum = pair_bev_iou.sum()

    loss = torch.where(
        pair_count > 0, loss_sum / pair_count.clamp_min(1.0),
        src_logits.sum() * 0.0)
    safe_pairs = pair_count.clamp_min(1.0)
    safe_inverted = inverted_count.clamp_min(1.0)
    safe_crossing = threshold_crossing_count.clamp_min(1.0)
    return {
        'loss_iou_classification_nms_rank': loss,
        'monitor_nms_rank_pair_count': pair_count.detach(),
        'monitor_nms_rank_pair_accuracy': (
            correct_count / safe_pairs).detach(),
        'monitor_nms_rank_weighted_accuracy': (
            weighted_correct / weight_sum.clamp_min(1e-12)).detach(),
        'monitor_nms_rank_inversion_count': inverted_count.detach(),
        'monitor_nms_rank_iou_gap_mean': (
            iou_gap_sum / safe_pairs).detach(),
        'monitor_nms_rank_inverted_iou_gap_mean': (
            inverted_gap_sum / safe_inverted).detach(),
        'monitor_nms_rank_bev_iou_mean': (
            bev_iou_sum / safe_pairs).detach(),
        'monitor_nms_rank_cross_0_7_pair_count': (
            threshold_crossing_count.detach()),
        'monitor_nms_rank_cross_0_7_wrong_fraction': (
            threshold_crossing_wrong / safe_crossing).detach(),
    }
