"""Exact, sparse pairwise 3D-IoU cost for Hungarian assignment."""
from __future__ import annotations

import torch
from torch import Tensor

from lib.losses.asymmetric_interval_depth_loss import (
    _centers_on_projected_rays,
    _decode_alpha,
    _decode_yaw_like_public_decoder,
    _inside_precomputed,
    _intersection_area_from_corners,
    _rectangle_corners,
)


@torch.no_grad()
def pairwise_iou3d_match_cost(
        outputs: dict[str, Tensor], target: dict[str, Tensor],
        decode_mean_sizes: Tensor) -> tuple[Tensor, dict[str, int]]:
    """Return exact ``[queries, targets]`` IoU with a sparse broad phase.

    Query and target corners/trigonometric terms are built once.  Exact
    polygon clipping runs only for pairs whose BEV AABBs and vertical spans
    overlap, because every other pair has provably zero 3D IoU.
    """
    boxes = outputs['pred_boxes']
    query_count = boxes.shape[0]
    target_count = target['labels'].shape[0]
    result = boxes.new_zeros((query_count, target_count))
    if query_count == 0 or target_count == 0:
        return result, {
            'pair_count': query_count * target_count,
            'exact_pair_count': 0,
        }

    device, dtype = boxes.device, boxes.dtype
    input_size = target['projective_input_size'].to(
        device=device, dtype=dtype)
    effective_calib = target['projective_image_effective_calib'].to(
        device=device, dtype=dtype)

    pred_uv = boxes[:, :2] * input_size
    pred_depth = outputs['pred_depth'][:, 0]
    pred_center, pred_ray_valid = _centers_on_projected_rays(
        pred_uv, pred_depth[:, None],
        effective_calib.expand(query_count, -1, -1))
    means = decode_mean_sizes.to(device=device, dtype=dtype)
    pred_dimensions = (
        outputs['pred_3d_dim'][:, None, :]
        + means[target['labels'].long()][None, :, :]).clamp_min(0.05)

    pred_alpha = _decode_alpha(outputs['pred_angle'])
    image_width = target['img_size'].to(device=device, dtype=dtype)[0]
    heading_calib = target['calibs'][0].to(device=device, dtype=dtype)
    pred_yaw = _decode_yaw_like_public_decoder(
        pred_alpha, boxes, image_width.expand(query_count),
        heading_calib.expand(query_count, -1, -1))

    target_uv = target['boxes_3d'][:, :2].to(dtype=dtype) * input_size
    target_depth = (
        target['depth'].reshape(-1).to(dtype=dtype)
        / target['depth_unit_scale'].reshape(-1).to(dtype=dtype).clamp_min(1e-6))
    target_center, target_ray_valid = _centers_on_projected_rays(
        target_uv, target_depth[:, None],
        effective_calib.expand(target_count, -1, -1))
    target_dimensions = target['src_size_3d'].to(dtype=dtype).clamp_min(0.05)
    target_yaw = target['projective_rotation_y'].reshape(-1).to(dtype=dtype)

    # A query's decoded dimensions depend on the candidate GT class.  KITTI
    # Car-only training makes these identical across targets, while keeping
    # this implementation correct for mixed-class matching.
    pred_h, pred_w, pred_l = pred_dimensions.unbind(-1)
    target_h, target_w, target_l = target_dimensions.unbind(-1)
    pred_cos, pred_sin = pred_yaw.cos(), pred_yaw.sin()
    target_cos, target_sin = target_yaw.cos(), target_yaw.sin()

    # Corners are class-dependent through dimensions, hence [Q,M,4,2].
    zero_center = pred_center.new_zeros((query_count, target_count, 2))
    pred_corners = _rectangle_corners(
        zero_center, pred_w, pred_l,
        pred_yaw[:, None].expand(-1, target_count))
    pred_corners = pred_corners + pred_center[:, None, None, (0, 2)]
    target_corners = _rectangle_corners(
        target_center[:, (0, 2)], target_w, target_l, target_yaw)

    pred_min, pred_max = pred_corners.amin(-2), pred_corners.amax(-2)
    target_min = target_corners.amin(-2)[None, :, :]
    target_max = target_corners.amax(-2)[None, :, :]
    broad_bev = ((pred_max[..., 0] >= target_min[..., 0])
                 & (target_max[..., 0] >= pred_min[..., 0])
                 & (pred_max[..., 1] >= target_min[..., 1])
                 & (target_max[..., 1] >= pred_min[..., 1]))
    pred_low = pred_center[:, None, 1] - pred_h * 0.5
    pred_high = pred_center[:, None, 1] + pred_h * 0.5
    target_low = target_center[None, :, 1] - target_h[None, :] * 0.5
    target_high = target_center[None, :, 1] + target_h[None, :] * 0.5
    vertical = (torch.minimum(pred_high, target_high)
                - torch.maximum(pred_low, target_low)).clamp_min(0)
    broad = (broad_bev & (vertical > 0)
             & pred_ray_valid[:, None] & target_ray_valid[None, :]
             & torch.isfinite(pred_dimensions).all(-1)
             & torch.isfinite(target_dimensions).all(-1)[None, :])

    candidate = broad.nonzero(as_tuple=False)
    if candidate.numel() == 0:
        return result, {
            'pair_count': query_count * target_count,
            'exact_pair_count': 0,
        }
    query_index, target_index = candidate.unbind(-1)
    first = pred_corners[query_index, target_index]
    second = target_corners[target_index]
    first_center_xz = pred_center[query_index][:, (0, 2)]
    second_center_xz = target_center[target_index][:, (0, 2)]
    first_valid = _inside_precomputed(
        first, second_center_xz,
        target_w[target_index] * 0.5,
        target_l[target_index] * 0.5,
        target_cos[target_index], target_sin[target_index])
    second_valid = _inside_precomputed(
        second, first_center_xz,
        pred_w[query_index, target_index] * 0.5,
        pred_l[query_index, target_index] * 0.5,
        pred_cos[query_index], pred_sin[query_index])
    bev_intersection = _intersection_area_from_corners(
        first, second, first_valid, second_valid)
    intersection = bev_intersection * vertical[query_index, target_index]
    pred_volume = (pred_h * pred_w * pred_l)[query_index, target_index]
    target_volume = (target_h * target_w * target_l)[target_index]
    union = pred_volume + target_volume - intersection
    values = intersection / union.clamp_min(torch.finfo(dtype).eps)
    result[query_index, target_index] = values.clamp(0, 1)
    return result, {
        'pair_count': query_count * target_count,
        'exact_pair_count': int(candidate.shape[0]),
    }
