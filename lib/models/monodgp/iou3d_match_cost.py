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
    if 'physical_ray_heading' in target:
        pred_yaw = torch.remainder(
            pred_alpha + torch.atan2(pred_center[:, 0], pred_center[:, 2])
            + torch.pi, 2.0 * torch.pi) - torch.pi
    else:
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


@torch.no_grad()
def batched_pairwise_iou3d_match_cost(
        outputs: dict[str, Tensor], targets: list[dict[str, Tensor]],
        decode_mean_sizes: Tensor) -> tuple[Tensor, dict[str, int]]:
    """Evaluate the legacy exact 3D-IoU program over a padded batch."""
    boxes = outputs['pred_boxes']
    batch_size, query_count = boxes.shape[:2]
    sizes = tuple(int(target['labels'].shape[0]) for target in targets)
    if len(sizes) != batch_size:
        raise ValueError('3D IoU target batch size does not match outputs')
    max_targets = max(sizes, default=0)
    result = boxes.new_zeros((batch_size, query_count, max_targets))
    pair_count = query_count * sum(sizes)
    if batch_size == 0 or query_count == 0 or max_targets == 0:
        return result, {'pair_count': pair_count, 'exact_pair_count': 0}

    device, dtype = boxes.device, boxes.dtype

    def pad(field, *, value=0.0, target_dtype=None):
        tensors = [target[field].to(
            device=device,
            dtype=(target_dtype if target_dtype is not None else dtype))
            for target in targets]
        return torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=value)

    target_labels = pad('labels', value=0, target_dtype=torch.long)
    target_boxes = pad('boxes_3d')
    target_depth = pad('depth').reshape(batch_size, max_targets)
    target_depth_scale = pad('depth_unit_scale').reshape(
        batch_size, max_targets)
    target_dimensions = pad('src_size_3d').clamp_min(0.05)
    target_yaw = pad('projective_rotation_y').reshape(
        batch_size, max_targets)
    input_size = torch.stack(tuple(
        target['projective_input_size'].to(device=device, dtype=dtype)
        for target in targets))
    effective_calib = torch.stack(tuple(
        target['projective_image_effective_calib'].to(
            device=device, dtype=dtype) for target in targets))
    target_valid = (
        torch.arange(max_targets, device=device)[None, :]
        < torch.as_tensor(sizes, device=device)[:, None])

    pred_uv = boxes[..., :2] * input_size[:, None, :]
    pred_depth = outputs['pred_depth'][..., 0]
    pred_center, pred_ray_valid = _centers_on_projected_rays(
        pred_uv, pred_depth[..., None],
        effective_calib[:, None, :, :].expand(
            -1, query_count, -1, -1))
    means = decode_mean_sizes.to(device=device, dtype=dtype)
    pred_dimensions = (
        outputs['pred_3d_dim'][:, :, None, :]
        + means[target_labels][:, None, :, :]).clamp_min(0.05)

    pred_alpha = _decode_alpha(outputs['pred_angle'])
    physical_heading = torch.as_tensor(
        ['physical_ray_heading' in target for target in targets],
        device=device, dtype=torch.bool)
    physical_yaw = torch.remainder(
        pred_alpha + torch.atan2(pred_center[..., 0], pred_center[..., 2])
        + torch.pi, 2.0 * torch.pi) - torch.pi
    if bool(physical_heading.all()):
        pred_yaw = physical_yaw
    else:
        image_width = torch.stack(tuple(
            target['img_size'].to(device=device, dtype=dtype)[0]
            for target in targets))
        heading_calib = torch.stack(tuple(
            (target['calibs'][0] if len(target['calibs'])
             else target['projective_image_effective_calib']).to(
                device=device, dtype=dtype) for target in targets))
        public_yaw = _decode_yaw_like_public_decoder(
            pred_alpha, boxes,
            image_width[:, None].expand(-1, query_count),
            heading_calib[:, None, :, :].expand(
                -1, query_count, -1, -1))
        pred_yaw = torch.where(
            physical_heading[:, None], physical_yaw, public_yaw)

    target_uv = target_boxes[..., :2] * input_size[:, None, :]
    target_depth_m = target_depth / target_depth_scale.clamp_min(1e-6)
    target_center, target_ray_valid = _centers_on_projected_rays(
        target_uv, target_depth_m[..., None],
        effective_calib[:, None, :, :].expand(
            -1, max_targets, -1, -1))

    pred_h, pred_w, pred_l = pred_dimensions.unbind(-1)
    target_h, target_w, target_l = target_dimensions.unbind(-1)
    pred_cos, pred_sin = pred_yaw.cos(), pred_yaw.sin()
    target_cos, target_sin = target_yaw.cos(), target_yaw.sin()

    zero_center = pred_center.new_zeros(
        (batch_size, query_count, max_targets, 2))
    pred_corners = _rectangle_corners(
        zero_center, pred_w, pred_l,
        pred_yaw[:, :, None].expand(-1, -1, max_targets))
    pred_corners = (
        pred_corners
        + pred_center[..., (0, 2)][:, :, None, None, :])
    target_corners = _rectangle_corners(
        target_center[..., (0, 2)], target_w, target_l, target_yaw)

    pred_min, pred_max = pred_corners.amin(-2), pred_corners.amax(-2)
    target_min = target_corners.amin(-2)[:, None, :, :]
    target_max = target_corners.amax(-2)[:, None, :, :]
    broad_bev = ((pred_max[..., 0] >= target_min[..., 0])
                 & (target_max[..., 0] >= pred_min[..., 0])
                 & (pred_max[..., 1] >= target_min[..., 1])
                 & (target_max[..., 1] >= pred_min[..., 1]))
    pred_low = pred_center[:, :, None, 1] - pred_h * 0.5
    pred_high = pred_center[:, :, None, 1] + pred_h * 0.5
    target_low = target_center[:, None, :, 1] - target_h[:, None, :] * 0.5
    target_high = target_center[:, None, :, 1] + target_h[:, None, :] * 0.5
    vertical = (torch.minimum(pred_high, target_high)
                - torch.maximum(pred_low, target_low)).clamp_min(0)
    broad = (broad_bev & (vertical > 0)
             & pred_ray_valid[:, :, None]
             & target_ray_valid[:, None, :]
             & torch.isfinite(pred_dimensions).all(-1)
             & torch.isfinite(target_dimensions).all(-1)[:, None, :]
             & target_valid[:, None, :])

    candidate = broad.nonzero(as_tuple=False)
    if candidate.numel() == 0:
        return result, {'pair_count': pair_count, 'exact_pair_count': 0}
    batch_index, query_index, target_index = candidate.unbind(-1)
    first = pred_corners[batch_index, query_index, target_index]
    second = target_corners[batch_index, target_index]
    first_center_xz = pred_center[
        batch_index, query_index][..., (0, 2)]
    second_center_xz = target_center[
        batch_index, target_index][..., (0, 2)]
    first_valid = _inside_precomputed(
        first, second_center_xz,
        target_w[batch_index, target_index] * 0.5,
        target_l[batch_index, target_index] * 0.5,
        target_cos[batch_index, target_index],
        target_sin[batch_index, target_index])
    second_valid = _inside_precomputed(
        second, first_center_xz,
        pred_w[batch_index, query_index, target_index] * 0.5,
        pred_l[batch_index, query_index, target_index] * 0.5,
        pred_cos[batch_index, query_index],
        pred_sin[batch_index, query_index])
    bev_intersection = _intersection_area_from_corners(
        first, second, first_valid, second_valid)
    intersection = bev_intersection * vertical[
        batch_index, query_index, target_index]
    pred_volume = (pred_h * pred_w * pred_l)[
        batch_index, query_index, target_index]
    target_volume = (target_h * target_w * target_l)[
        batch_index, target_index]
    union = pred_volume + target_volume - intersection
    values = intersection / union.clamp_min(torch.finfo(dtype).eps)
    result[batch_index, query_index, target_index] = values.clamp(0, 1)
    return result, {
        'pair_count': pair_count,
        'exact_pair_count': int(candidate.shape[0]),
    }
