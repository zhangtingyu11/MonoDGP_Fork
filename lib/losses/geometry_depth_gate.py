"""Detached geometric quality weights for final object-depth gradients.

This is gradient modulation, not a new probability likelihood. Only the full-P2
physical-heading target contract is supported; missing metadata is an error.
"""
import math
import torch

from lib.losses.asymmetric_interval_depth_loss import (
    _centers_on_projected_rays, _decode_alpha, _matched,
    _matched_batch_indices, paired_iou3d,
)


def validate_gate_config(config):
    if config.get('mode', 'geometry') not in ('geometry', 'uniform_mean'):
        raise ValueError('Unknown depth gradient gate mode')
    low = float(config.get('iou_start', .7))
    high = float(config.get('iou_full', .75))
    floor = float(config.get('minimum_gradient', .5))
    if not (0 < low < high <= 1 and 0 < floor <= 1):
        raise ValueError('Require 0 < iou_start < iou_full <= 1 and 0 < minimum_gradient <= 1')
    return low, high, floor


def depth_gradient_proxy(mean, weights):
    if mean.shape != weights.shape:
        raise ValueError('Depth gradient weights must have the matched mean shape')
    # Forward-identical for finite inputs; weights cannot receive gradients.
    return mean.detach() + weights.detach() * (mean - mean.detach())


@torch.no_grad()
def matched_geometry_depth_weights(outputs, targets, indices, config):
    low, high, floor = validate_gate_config(config)
    if not all('physical_ray_heading' in t for t in targets):
        raise ValueError('Geometry depth gate requires physical_ray_heading/full-P2 metadata')
    device = outputs['pred_depth'].device
    dtype = outputs['pred_depth'].dtype
    heading_flags = torch.stack([
        torch.as_tensor(t['physical_ray_heading'], device=device, dtype=torch.bool)
        for t in targets])
    if not heading_flags.all():
        raise ValueError('Geometry depth gate requires physical-ray heading targets')
    batch = _matched_batch_indices(indices, device)
    query = torch.cat([torch.as_tensor(i, device=device, dtype=torch.long)
                       for i, _ in indices])
    pred = outputs['pred_depth'][batch, query]
    mean = pred[:, 0]
    weights = torch.ones_like(mean)
    labels = _matched(targets, indices, 'labels', device).reshape(-1)
    scale = _matched(targets, indices, 'depth_unit_scale', device, dtype).reshape(-1)
    gt_virtual = _matched(targets, indices, 'depth', device, dtype).reshape(-1)
    if not (torch.isfinite(pred).all() and torch.isfinite(scale).all()
            and (scale > 0).all() and torch.isfinite(gt_virtual).all()):
        raise FloatingPointError('Nonfinite depth or invalid depth-unit scale in geometry gate')
    gt_z = gt_virtual / scale
    pred_z = mean / scale
    boxes = outputs['pred_boxes'][batch, query]
    pred_dim = outputs['pred_3d_dim'][batch, query]
    means = torch.as_tensor(config.get('decode_mean_sizes', [[0., 0., 0.]] * 3),
                            device=device, dtype=dtype)
    pred_labels = outputs['pred_logits'][batch, query].argmax(-1)
    pred_dim = pred_dim + means[pred_labels]
    gt_dim = _matched(targets, indices, 'src_size_3d', device, dtype)
    alpha = _decode_alpha(outputs['pred_angle'][batch, query])
    gt_yaw = _matched(targets, indices, 'projective_rotation_y', device, dtype).reshape(-1)
    sizes = torch.stack([t['projective_input_size'].to(device=device, dtype=dtype)
                         for t in targets])[batch]
    calibs = torch.stack([t['projective_image_effective_calib'].to(device=device, dtype=dtype)
                          for t in targets])[batch]
    uv = boxes[:, :2] * sizes
    gt_uv = _matched(targets, indices, 'boxes_3d', device, dtype)[:, :2] * sizes
    center, ray_ok = _centers_on_projected_rays(uv, pred_z[:, None], calibs)
    anchor, anchor_ok = _centers_on_projected_rays(uv, gt_z[:, None], calibs)
    gt_center, gt_ok = _centers_on_projected_rays(gt_uv, gt_z[:, None], calibs)
    yaw = torch.remainder(alpha + torch.atan2(center[:, 0], center[:, 2]) + math.pi,
                          2 * math.pi) - math.pi
    anchor_yaw = torch.remainder(alpha + torch.atan2(anchor[:, 0], anchor[:, 2]) + math.pi,
                                 2 * math.pi) - math.pi
    finite = (torch.isfinite(pred_dim).all(-1) & torch.isfinite(gt_dim).all(-1)
              & torch.isfinite(center).all(-1) & torch.isfinite(anchor).all(-1)
              & torch.isfinite(gt_center).all(-1) & torch.isfinite(yaw)
              & torch.isfinite(anchor_yaw) & torch.isfinite(gt_yaw))
    eligible = (labels.eq(int(config.get('car_class_id', 1))) & finite
                & (pred_dim > 0).all(-1) & (gt_dim > 0).all(-1)
                & (pred_z > 0) & (gt_z > 2) & (gt_z < 65)
                & ray_ok & anchor_ok & gt_ok)
    selected = torch.nonzero(eligible).flatten()
    current_iou = torch.zeros_like(mean)
    anchor_iou = torch.zeros_like(mean)
    if selected.numel():
        # Translation to the GT origin reduces large-coordinate cancellation.
        origin = gt_center[selected]
        zeros = torch.zeros_like(origin)
        values = paired_iou3d(
            torch.cat((center[selected] - origin, anchor[selected] - origin)),
            pred_dim[selected].repeat(2, 1),
            torch.cat((yaw[selected], anchor_yaw[selected])),
            zeros.repeat(2, 1), gt_dim[selected].repeat(2, 1),
            gt_yaw[selected].repeat(2))
        current_iou[selected], anchor_iou[selected] = values.chunk(2)
    if not (torch.isfinite(current_iou).all() and torch.isfinite(anchor_iou).all()):
        raise FloatingPointError('Nonfinite IoU in geometry depth gate')
    supported = eligible & anchor_iou.ge(low)
    reduction = ((current_iou - low) / (high - low)).clamp(0, 1)
    weights = torch.where(supported, 1 - (1 - floor) * reduction, weights)
    geometric_weights = weights
    if config.get('mode', 'geometry') == 'uniform_mean' and weights.numel():
        # Mechanism control: same batch-average multiplier on every match.
        # This does not conserve output-gradient energy or optimizer updates.
        weights = weights.mean().expand_as(weights)
    return weights, {
        'weights': weights, 'eligible': eligible, 'supported': supported,
        'geometric_weights': geometric_weights,
        'current_iou': current_iou, 'anchor_iou': anchor_iou,
        'source_batch': batch, 'source_query': query,
        'predicted_physical_depth': pred_z, 'gt_physical_depth': gt_z,
    }
