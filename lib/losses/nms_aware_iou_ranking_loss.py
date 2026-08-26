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
        group_num: int, bev_iou_threshold: float,
        require_same_target: bool = True,
        strict_threshold: bool = False,
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
    if require_same_target:
        same_entity = (
            grouped_target[..., :, None] == grouped_target[..., None, :])
    else:
        grouped_labels = assigned_labels.reshape(
            batch_size, group_num, queries_per_group)
        same_entity = (
            grouped_labels[..., :, None] == grouped_labels[..., None, :])
    valid = (
        upper
        & same_entity
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
    triggered = (
        bev_iou > bev_iou_threshold if strict_threshold
        else bev_iou >= bev_iou_threshold)
    return (
        batch[triggered], first[triggered], second[triggered],
        bev_iou[triggered])


@torch.no_grad()
def _hungarian_matched_unmatched_pairs_batched(
        outputs: dict[str, Tensor], targets: list[dict[str, Tensor]],
        iou3d_matrix: Tensor,
        iou3d_only_indices: list[tuple[Tensor, Tensor]],
        decode_mean_sizes: Tensor, group_num: int,
        bev_iou_threshold: float,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return only pure-IoU matched-to-unmatched pairs above BEV threshold.

    The expensive exact 3D-IoU matrix and the pure-IoU Hungarian assignment
    are produced by the matcher.  This function expands only each matched
    anchor against the 50 queries in its own independent group, removes every
    Hungarian-matched query, applies an AABB broad phase, and runs exact
    rotated intersection only for the surviving pairs.
    """
    batch_size, query_count = outputs['pred_logits'].shape[:2]
    if len(iou3d_only_indices) != batch_size:
        raise ValueError('pure-IoU Hungarian batch size does not match logits')
    queries_per_group = query_count // group_num
    if queries_per_group * group_num != query_count:
        raise ValueError('query count must be divisible by group_num')

    device = outputs['pred_logits'].device
    _, _, _, assigned_labels = _assign_queries_to_targets(
        iou3d_matrix, targets)
    decoded = _decode_query_bev_geometry_batched(
        outputs, targets, assigned_labels, decode_mean_sizes)
    decoded_corners, _, _, _, decoded_yaw, geometry_valid = decoded
    centers = decoded_corners.mean(dim=-2)

    matched_mask = torch.zeros(
        (batch_size, query_count), dtype=torch.bool, device=device)
    anchor_batch_parts = []
    anchor_query_parts = []
    anchor_target_parts = []
    for batch_index, (source, target_index) in enumerate(
            iou3d_only_indices):
        source = source.to(device=device, dtype=torch.long)
        target_index = target_index.to(device=device, dtype=torch.long)
        if source.numel() != target_index.numel():
            raise ValueError('pure-IoU Hungarian source/target sizes differ')
        if source.numel() == 0:
            continue
        target_count = int(targets[batch_index]['labels'].numel())
        if ((source < 0).any() or (source >= query_count).any()
                or (target_index < 0).any()
                or (target_index >= target_count).any()):
            raise ValueError('pure-IoU Hungarian index is out of range')
        if torch.unique(source).numel() != source.numel():
            raise ValueError('pure-IoU Hungarian sources must be unique')
        matched_mask[batch_index, source] = True
        anchor_batch_parts.append(torch.full_like(source, batch_index))
        anchor_query_parts.append(source)
        anchor_target_parts.append(target_index)

    if not anchor_query_parts:
        empty_index = torch.empty(0, device=device, dtype=torch.long)
        empty_value = outputs['pred_logits'].new_empty(0)
        return (empty_index, empty_index, empty_index, empty_index,
                empty_index, empty_value)

    anchor_batch = torch.cat(anchor_batch_parts)
    anchor_query = torch.cat(anchor_query_parts)
    anchor_target = torch.cat(anchor_target_parts)
    anchor_group = anchor_query.div(
        queries_per_group, rounding_mode='floor')
    local_query = torch.arange(
        queries_per_group, device=device, dtype=torch.long)
    candidate_query = (
        anchor_group[:, None] * queries_per_group + local_query[None, :])
    candidate_valid = ~matched_mask[
        anchor_batch[:, None], candidate_query]
    unit_id, candidate_local = candidate_valid.nonzero(as_tuple=True)
    candidate_query = candidate_query[unit_id, candidate_local]
    pair_batch = anchor_batch[unit_id]
    pair_anchor = anchor_query[unit_id]
    pair_target = anchor_target[unit_id]

    if pair_anchor.numel() == 0:
        empty_value = outputs['pred_logits'].new_empty(0)
        return (pair_batch, pair_anchor, candidate_query, pair_target,
                unit_id, empty_value)

    padded_labels = torch.zeros(
        (batch_size, iou3d_matrix.shape[-1]),
        device=device, dtype=torch.long)
    for batch_index, target in enumerate(targets):
        count = int(target['labels'].numel())
        padded_labels[batch_index, :count] = target['labels'].reshape(
            -1).to(device=device, dtype=torch.long)
    pair_label = padded_labels[pair_batch, pair_target]
    means = decode_mean_sizes.to(
        device=device, dtype=outputs['pred_3d_dim'].dtype)
    anchor_dimensions = (
        outputs['pred_3d_dim'][pair_batch, pair_anchor]
        + means[pair_label]).clamp_min(0.05)
    candidate_dimensions = (
        outputs['pred_3d_dim'][pair_batch, candidate_query]
        + means[pair_label]).clamp_min(0.05)
    _, anchor_width, anchor_length = anchor_dimensions.unbind(-1)
    _, candidate_width, candidate_length = candidate_dimensions.unbind(-1)
    anchor_corners = _rectangle_corners(
        centers[pair_batch, pair_anchor], anchor_width, anchor_length,
        decoded_yaw[pair_batch, pair_anchor])
    candidate_corners = _rectangle_corners(
        centers[pair_batch, candidate_query], candidate_width,
        candidate_length, decoded_yaw[pair_batch, candidate_query])
    anchor_min, anchor_max = (
        anchor_corners.amin(dim=-2), anchor_corners.amax(dim=-2))
    candidate_min, candidate_max = (
        candidate_corners.amin(dim=-2), candidate_corners.amax(dim=-2))
    broad = (
        geometry_valid[pair_batch, pair_anchor]
        & geometry_valid[pair_batch, candidate_query]
        & torch.isfinite(anchor_dimensions).all(-1)
        & torch.isfinite(candidate_dimensions).all(-1)
        & (anchor_max[:, 0] >= candidate_min[:, 0])
        & (candidate_max[:, 0] >= anchor_min[:, 0])
        & (anchor_max[:, 1] >= candidate_min[:, 1])
        & (candidate_max[:, 1] >= anchor_min[:, 1]))
    broad_index = broad.nonzero(as_tuple=True)[0]
    pair_batch = pair_batch[broad_index]
    pair_anchor = pair_anchor[broad_index]
    candidate_query = candidate_query[broad_index]
    pair_target = pair_target[broad_index]
    unit_id = unit_id[broad_index]
    anchor_corners = anchor_corners[broad_index]
    candidate_corners = candidate_corners[broad_index]
    anchor_width = anchor_width[broad_index]
    anchor_length = anchor_length[broad_index]
    candidate_width = candidate_width[broad_index]
    candidate_length = candidate_length[broad_index]
    anchor_yaw = decoded_yaw[pair_batch, pair_anchor]
    candidate_yaw = decoded_yaw[pair_batch, candidate_query]

    anchor_inside = _inside_precomputed(
        anchor_corners, candidate_corners.mean(dim=-2),
        candidate_width * 0.5, candidate_length * 0.5,
        candidate_yaw.cos(), candidate_yaw.sin())
    candidate_inside = _inside_precomputed(
        candidate_corners, anchor_corners.mean(dim=-2),
        anchor_width * 0.5, anchor_length * 0.5,
        anchor_yaw.cos(), anchor_yaw.sin())
    intersection = _intersection_area_from_corners(
        anchor_corners, candidate_corners,
        anchor_inside, candidate_inside)
    anchor_area = anchor_width * anchor_length
    candidate_area = candidate_width * candidate_length
    union = anchor_area + candidate_area - intersection
    bev_iou = intersection / union.clamp_min(
        torch.finfo(intersection.dtype).eps)
    keep = bev_iou > bev_iou_threshold
    return (pair_batch[keep], pair_anchor[keep],
            candidate_query[keep], pair_target[keep], unit_id[keep],
            bev_iou[keep])


@torch.no_grad()
def _assign_queries_to_targets(
        iou3d_matrix: Tensor, targets: list[dict[str, Tensor]],
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return target counts, nearest-target IoU/index and assigned labels."""
    device = iou3d_matrix.device
    target_slots = iou3d_matrix.shape[-1]
    target_counts = torch.as_tensor(
        [target['labels'].numel() for target in targets],
        device=device, dtype=torch.long)
    if target_slots == 0:
        empty_shape = iou3d_matrix.shape[:2]
        return (
            target_counts,
            iou3d_matrix.new_zeros(empty_shape),
            torch.zeros(empty_shape, device=device, dtype=torch.long),
            torch.zeros(empty_shape, device=device, dtype=torch.long))
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
    return target_counts, max_iou, assigned_target, assigned_labels


def _all_conflicting_pair_ranking_loss(
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
            'monitor_nms_rank_pair_correct_count': zero.detach(),
            'monitor_nms_rank_pair_accuracy': zero.detach(),
            'monitor_nms_rank_weight_sum': zero.detach(),
            'monitor_nms_rank_weighted_correct_sum': zero.detach(),
            'monitor_nms_rank_weighted_accuracy': zero.detach(),
            'monitor_nms_rank_inversion_count': zero.detach(),
            'monitor_nms_rank_iou_gap_sum': zero.detach(),
            'monitor_nms_rank_iou_gap_mean': zero.detach(),
            'monitor_nms_rank_inverted_iou_gap_sum': zero.detach(),
            'monitor_nms_rank_inverted_iou_gap_mean': zero.detach(),
            'monitor_nms_rank_bev_iou_sum': zero.detach(),
            'monitor_nms_rank_bev_iou_mean': zero.detach(),
            'monitor_nms_rank_cross_0_7_pair_count': zero.detach(),
            'monitor_nms_rank_cross_0_7_wrong_count': zero.detach(),
            'monitor_nms_rank_cross_0_7_wrong_fraction': zero.detach(),
        }
    target_counts, max_iou, assigned_target, assigned_labels = (
        _assign_queries_to_targets(iou3d_matrix, targets))
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
        'monitor_nms_rank_pair_correct_count': correct_count.detach(),
        'monitor_nms_rank_pair_accuracy': (
            correct_count / safe_pairs).detach(),
        'monitor_nms_rank_weight_sum': weight_sum.detach(),
        'monitor_nms_rank_weighted_correct_sum': weighted_correct.detach(),
        'monitor_nms_rank_weighted_accuracy': (
            weighted_correct / weight_sum.clamp_min(1e-12)).detach(),
        'monitor_nms_rank_inversion_count': inverted_count.detach(),
        'monitor_nms_rank_iou_gap_sum': iou_gap_sum.detach(),
        'monitor_nms_rank_iou_gap_mean': (
            iou_gap_sum / safe_pairs).detach(),
        'monitor_nms_rank_inverted_iou_gap_sum': inverted_gap_sum.detach(),
        'monitor_nms_rank_inverted_iou_gap_mean': (
            inverted_gap_sum / safe_inverted).detach(),
        'monitor_nms_rank_bev_iou_sum': bev_iou_sum.detach(),
        'monitor_nms_rank_bev_iou_mean': (
            bev_iou_sum / safe_pairs).detach(),
        'monitor_nms_rank_cross_0_7_pair_count': (
            threshold_crossing_count.detach()),
        'monitor_nms_rank_cross_0_7_wrong_count': (
            threshold_crossing_wrong.detach()),
        'monitor_nms_rank_cross_0_7_wrong_fraction': (
            threshold_crossing_wrong / safe_crossing).detach(),
    }


def _best_query_suppressor_ranking_loss(
        src_logits: Tensor, outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]], iou3d_matrix: Tensor,
        decode_mean_sizes: Tensor, group_num: int,
        bev_iou_threshold: float, min_iou_delta: float,
        ) -> dict[str, Tensor]:
    """Penalize only candidates currently able to outrank each GT's best box.

    A training unit is one (image, independent query group, GT) tuple. The
    highest exact-3D-IoU query for that GT is compared only with same-class
    predicted boxes whose BEV overlap exceeds the deployment NMS threshold.
    Loss is computed for currently inverted pairs, averaged first within each
    GT unit and then across active units. Pair selection and IoU weights are
    detached; gradients flow only through the class logits.
    """
    target_slots = iou3d_matrix.shape[-1]
    zero = src_logits.sum() * 0.0
    zero_detached = zero.detach()
    empty = {
        'loss_iou_classification_nms_rank': zero,
        'monitor_nms_rank_pair_count': zero_detached,
        'monitor_nms_rank_pair_correct_count': zero_detached,
        'monitor_nms_rank_pair_accuracy': zero_detached,
        'monitor_nms_rank_weight_sum': zero_detached,
        'monitor_nms_rank_weighted_correct_sum': zero_detached,
        'monitor_nms_rank_weighted_accuracy': zero_detached,
        'monitor_nms_rank_inversion_count': zero_detached,
        'monitor_nms_rank_iou_gap_sum': zero_detached,
        'monitor_nms_rank_iou_gap_mean': zero_detached,
        'monitor_nms_rank_inverted_iou_gap_sum': zero_detached,
        'monitor_nms_rank_inverted_iou_gap_mean': zero_detached,
        'monitor_nms_rank_bev_iou_sum': zero_detached,
        'monitor_nms_rank_bev_iou_mean': zero_detached,
        'monitor_nms_rank_cross_0_7_pair_count': zero_detached,
        'monitor_nms_rank_cross_0_7_wrong_count': zero_detached,
        'monitor_nms_rank_cross_0_7_wrong_fraction': zero_detached,
        'monitor_nms_rank_gt_unit_count': zero_detached,
        'monitor_nms_rank_optimized_gt_unit_count': zero_detached,
    }
    if target_slots == 0:
        return empty

    batch_size, query_count = src_logits.shape[:2]
    queries_per_group = query_count // group_num
    target_counts, _, assigned_target, assigned_labels = (
        _assign_queries_to_targets(iou3d_matrix, targets))
    batch, first, second, pair_bev_iou = _triggered_nms_pairs_batched(
        outputs, targets, assigned_target, assigned_labels, target_counts,
        decode_mean_sizes=decode_mean_sizes, group_num=group_num,
        bev_iou_threshold=bev_iou_threshold,
        require_same_target=False, strict_threshold=True)
    if first.numel() == 0:
        return empty

    device = src_logits.device
    detached_iou = iou3d_matrix.detach().clamp(0, 1)
    padded_labels = torch.zeros(
        batch_size, target_slots, device=device, dtype=torch.long)
    for batch_index, target in enumerate(targets):
        count = int(target['labels'].numel())
        padded_labels[batch_index, :count] = target['labels'].reshape(
            -1).to(device=device, dtype=torch.long)

    grouped_iou = detached_iou.reshape(
        batch_size, group_num, queries_per_group, target_slots)
    grouped_assigned_labels = assigned_labels.reshape(
        batch_size, group_num, queries_per_group)
    compatible = (
        grouped_assigned_labels[..., None]
        == padded_labels[:, None, None, :])
    target_slot = torch.arange(target_slots, device=device)
    valid_target = target_slot[None, None, :] < target_counts[:, None, None]
    candidate_iou = torch.where(
        compatible & valid_target[:, :, None, :], grouped_iou,
        torch.full_like(grouped_iou, -1.0))
    best_iou, best_local = candidate_iou.max(dim=2)
    valid_unit = valid_target.expand(-1, group_num, -1) & (best_iou >= 0)

    pair_group = first.div(queries_per_group, rounding_mode='floor')
    first_local = first.remainder(queries_per_group)
    second_local = second.remainder(queries_per_group)
    pair_best_local = best_local[batch, pair_group]
    first_is_best = first_local[:, None] == pair_best_local
    second_is_best = second_local[:, None] == pair_best_local
    touches_best = first_is_best ^ second_is_best
    candidate = torch.where(
        first_is_best, second[:, None], first[:, None])
    pair_target = target_slot[None, :].expand(first.numel(), -1)
    pair_batch = batch[:, None].expand_as(pair_target)
    candidate_true_iou = detached_iou[
        pair_batch, candidate, pair_target]
    iou_gap = best_iou[batch, pair_group] - candidate_true_iou
    valid_pair = (
        touches_best
        & valid_unit[batch, pair_group]
        & (iou_gap > min_iou_delta))
    pair_index, target_index = valid_pair.nonzero(as_tuple=True)
    if pair_index.numel() == 0:
        result = dict(empty)
        result['monitor_nms_rank_gt_unit_count'] = (
            valid_unit.sum().to(src_logits.dtype).detach())
        return result

    selected_batch = batch[pair_index]
    selected_group = pair_group[pair_index]
    selected_best = (
        selected_group * queries_per_group
        + best_local[selected_batch, selected_group, target_index])
    selected_candidate = candidate[pair_index, target_index]
    selected_labels = padded_labels[selected_batch, target_index]
    best_score = src_logits[
        selected_batch, selected_best, selected_labels]
    candidate_score = src_logits[
        selected_batch, selected_candidate, selected_labels]
    signed_margin = best_score - candidate_score
    weights = iou_gap[pair_index, target_index]
    correct = signed_margin > 0
    inverted = ~correct
    selected_bev_iou = pair_bev_iou[pair_index]
    best_true_iou = best_iou[
        selected_batch, selected_group, target_index]
    candidate_iou = detached_iou[
        selected_batch, selected_candidate, target_index]
    threshold_crossing = (
        (best_true_iou >= 0.7) & (candidate_iou < 0.7))

    inverted_loss = weights * F.softplus(-signed_margin)
    inverted_loss = inverted_loss * inverted.to(inverted_loss.dtype)
    unit_id = (
        (selected_batch * group_num + selected_group) * target_slots
        + target_index)
    unit_slots = batch_size * group_num * target_slots
    unit_loss_sum = src_logits.new_zeros(unit_slots)
    unit_inversion_count = src_logits.new_zeros(unit_slots)
    unit_loss_sum.scatter_add_(0, unit_id, inverted_loss)
    unit_inversion_count.scatter_add_(
        0, unit_id, inverted.to(src_logits.dtype))
    active_unit = unit_inversion_count > 0
    per_unit_loss = unit_loss_sum / unit_inversion_count.clamp_min(1)
    active_unit_count = active_unit.sum().to(src_logits.dtype)
    loss = per_unit_loss.sum() / active_unit_count.clamp_min(1)

    pair_count = src_logits.new_tensor(float(pair_index.numel()))
    correct_count = correct.sum().to(src_logits.dtype)
    inversion_count = inverted.sum().to(src_logits.dtype)
    weight_sum = weights.sum()
    weighted_correct = weights[correct].sum()
    iou_gap_sum = weights.sum()
    inverted_gap_sum = weights[inverted].sum()
    crossing_count = threshold_crossing.sum().to(src_logits.dtype)
    crossing_wrong = (
        threshold_crossing & inverted).sum().to(src_logits.dtype)
    safe_pairs = pair_count.clamp_min(1)
    return {
        'loss_iou_classification_nms_rank': loss,
        'monitor_nms_rank_pair_count': pair_count.detach(),
        'monitor_nms_rank_pair_correct_count': correct_count.detach(),
        'monitor_nms_rank_pair_accuracy': (
            correct_count / safe_pairs).detach(),
        'monitor_nms_rank_weight_sum': weight_sum.detach(),
        'monitor_nms_rank_weighted_correct_sum': weighted_correct.detach(),
        'monitor_nms_rank_weighted_accuracy': (
            weighted_correct / weight_sum.clamp_min(1e-12)).detach(),
        'monitor_nms_rank_inversion_count': inversion_count.detach(),
        'monitor_nms_rank_iou_gap_sum': iou_gap_sum.detach(),
        'monitor_nms_rank_iou_gap_mean': (
            iou_gap_sum / safe_pairs).detach(),
        'monitor_nms_rank_inverted_iou_gap_sum': inverted_gap_sum.detach(),
        'monitor_nms_rank_inverted_iou_gap_mean': (
            inverted_gap_sum / inversion_count.clamp_min(1)).detach(),
        'monitor_nms_rank_bev_iou_sum': selected_bev_iou.sum().detach(),
        'monitor_nms_rank_bev_iou_mean': (
            selected_bev_iou.sum() / safe_pairs).detach(),
        'monitor_nms_rank_cross_0_7_pair_count': crossing_count.detach(),
        'monitor_nms_rank_cross_0_7_wrong_count': crossing_wrong.detach(),
        'monitor_nms_rank_cross_0_7_wrong_fraction': (
            crossing_wrong / crossing_count.clamp_min(1)).detach(),
        'monitor_nms_rank_gt_unit_count': (
            valid_unit.sum().to(src_logits.dtype).detach()),
        'monitor_nms_rank_optimized_gt_unit_count': (
            active_unit_count.detach()),
    }


def _hungarian_unmatched_overlap_ranking_loss(
        src_logits: Tensor, outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]], iou3d_matrix: Tensor,
        iou3d_only_indices: list[tuple[Tensor, Tensor]],
        decode_mean_sizes: Tensor, group_num: int,
        bev_iou_threshold: float, **_: object,
        ) -> dict[str, Tensor]:
    """Pure RankNet from each pure-IoU match to overlapping unmatched boxes.

    One unit is a (sample, independent query group, GT) Hungarian match.  All
    unmatched queries in that group whose decoded BEV box overlaps the match
    above the deployment threshold participate, regardless of their current
    score.  Pair losses are averaged within a unit and then across active
    units, so crowded GTs do not dominate the batch.
    """
    zero = src_logits.sum() * 0.0
    zero_detached = zero.detach()
    empty = {
        'loss_iou_classification_nms_rank': zero,
        'monitor_nms_rank_pair_count': zero_detached,
        'monitor_nms_rank_pair_correct_count': zero_detached,
        'monitor_nms_rank_pair_accuracy': zero_detached,
        'monitor_nms_rank_inversion_count': zero_detached,
        'monitor_nms_rank_bev_iou_sum': zero_detached,
        'monitor_nms_rank_bev_iou_mean': zero_detached,
        'monitor_nms_rank_score_margin_sum': zero_detached,
        'monitor_nms_rank_score_margin_mean': zero_detached,
        'monitor_nms_rank_gt_unit_count': zero_detached,
        'monitor_nms_rank_optimized_gt_unit_count': zero_detached,
        'monitor_nms_rank_active_gt_fraction': zero_detached,
        'monitor_nms_rank_pair_per_active_gt': zero_detached,
        'monitor_nms_rank_matched_not_top_gt_unit_count': zero_detached,
        'monitor_nms_rank_matched_not_top_gt_fraction': zero_detached,
    }
    if iou3d_only_indices is None:
        raise RuntimeError(
            'Hungarian-unmatched ranking requires pure 3D-IoU indices')

    unit_count = sum(int(source.numel())
                     for source, _ in iou3d_only_indices)
    result = dict(empty)
    result['monitor_nms_rank_gt_unit_count'] = src_logits.new_tensor(
        float(unit_count)).detach()
    if unit_count == 0:
        return result

    (pair_batch, matched_query, unmatched_query, pair_target, unit_id,
     pair_bev_iou) = _hungarian_matched_unmatched_pairs_batched(
         outputs, targets, iou3d_matrix, iou3d_only_indices,
         decode_mean_sizes=decode_mean_sizes, group_num=group_num,
         bev_iou_threshold=bev_iou_threshold)
    if matched_query.numel() == 0:
        return result

    device = src_logits.device
    padded_labels = torch.zeros(
        (src_logits.shape[0], iou3d_matrix.shape[-1]),
        device=device, dtype=torch.long)
    for batch_index, target in enumerate(targets):
        count = int(target['labels'].numel())
        padded_labels[batch_index, :count] = target['labels'].reshape(
            -1).to(device=device, dtype=torch.long)
    pair_label = padded_labels[pair_batch, pair_target]
    matched_score = src_logits[pair_batch, matched_query, pair_label]
    unmatched_score = src_logits[pair_batch, unmatched_query, pair_label]
    score_margin = matched_score - unmatched_score
    pair_loss = F.softplus(-score_margin)
    correct = score_margin > 0
    incorrect = ~correct

    unit_loss_sum = src_logits.new_zeros(unit_count)
    unit_pair_count = src_logits.new_zeros(unit_count)
    unit_incorrect_count = src_logits.new_zeros(unit_count)
    unit_loss_sum.scatter_add_(0, unit_id, pair_loss)
    unit_pair_count.scatter_add_(
        0, unit_id, torch.ones_like(pair_loss))
    unit_incorrect_count.scatter_add_(
        0, unit_id, incorrect.to(src_logits.dtype))
    active_unit = unit_pair_count > 0
    active_unit_count = active_unit.sum().to(src_logits.dtype)
    per_unit_loss = unit_loss_sum / unit_pair_count.clamp_min(1)
    loss = per_unit_loss.sum() / active_unit_count.clamp_min(1)

    pair_count = src_logits.new_tensor(float(matched_query.numel()))
    correct_count = correct.sum().to(src_logits.dtype)
    inversion_count = incorrect.sum().to(src_logits.dtype)
    matched_not_top_count = (
        unit_incorrect_count > 0).sum().to(src_logits.dtype)
    safe_pairs = pair_count.clamp_min(1)
    safe_active_units = active_unit_count.clamp_min(1)
    result.update({
        'loss_iou_classification_nms_rank': loss,
        'monitor_nms_rank_pair_count': pair_count.detach(),
        'monitor_nms_rank_pair_correct_count': correct_count.detach(),
        'monitor_nms_rank_pair_accuracy': (
            correct_count / safe_pairs).detach(),
        'monitor_nms_rank_inversion_count': inversion_count.detach(),
        'monitor_nms_rank_bev_iou_sum': pair_bev_iou.sum().detach(),
        'monitor_nms_rank_bev_iou_mean': (
            pair_bev_iou.sum() / safe_pairs).detach(),
        'monitor_nms_rank_score_margin_sum': score_margin.sum().detach(),
        'monitor_nms_rank_score_margin_mean': (
            score_margin.sum() / safe_pairs).detach(),
        'monitor_nms_rank_optimized_gt_unit_count': (
            active_unit_count.detach()),
        'monitor_nms_rank_active_gt_fraction': (
            active_unit_count / max(unit_count, 1)).detach(),
        'monitor_nms_rank_pair_per_active_gt': (
            pair_count / safe_active_units).detach(),
        'monitor_nms_rank_matched_not_top_gt_unit_count': (
            matched_not_top_count.detach()),
        'monitor_nms_rank_matched_not_top_gt_fraction': (
            matched_not_top_count / safe_active_units).detach(),
    })
    return result


def nms_aware_iou_ranking_loss(
        src_logits: Tensor, outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]], iou3d_matrix: Tensor,
        decode_mean_sizes: Tensor, group_num: int,
        iou3d_only_indices: list[tuple[Tensor, Tensor]] | None = None,
        bev_iou_threshold: float = 0.8,
        min_iou_delta: float = 1e-6,
        strategy: str = 'all_conflicting_pairs') -> dict[str, Tensor]:
    """Dispatch one of the reproducible NMS-aware ranking objectives."""
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
    common = dict(
        src_logits=src_logits, outputs=outputs, targets=targets,
        iou3d_matrix=iou3d_matrix,
        iou3d_only_indices=iou3d_only_indices,
        decode_mean_sizes=decode_mean_sizes, group_num=group_num,
        bev_iou_threshold=bev_iou_threshold,
        min_iou_delta=min_iou_delta)
    if strategy == 'all_conflicting_pairs':
        common.pop('iou3d_only_indices')
        return _all_conflicting_pair_ranking_loss(**common)
    if strategy == 'best_query_suppressors':
        common.pop('iou3d_only_indices')
        return _best_query_suppressor_ranking_loss(**common)
    if strategy == 'hungarian_unmatched_overlap':
        return _hungarian_unmatched_overlap_ranking_loss(**common)
    raise ValueError(f'unsupported NMS ranking strategy: {strategy}')
