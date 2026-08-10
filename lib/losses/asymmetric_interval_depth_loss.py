"""GT-attributed asymmetric interval supervision for object depth.

The feasible interval follows experiment 14 literally: hold the predicted
dimensions, observation angle and projected center ray fixed, move only metric
depth, and retain depths whose 3D IoU with the matched GT box is at least the
configured threshold.  Interval construction is detached and training-only.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

LAPLACE_FACTOR = 1.4142


def _centers_on_projected_rays(
        centers_uv: Tensor, depths: Tensor,
        effective_calibs: Tensor) -> tuple[Tensor, Tensor]:
    u, v = centers_uv.unbind(-1)
    z = depths.squeeze(-1)
    p = effective_calibs
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
    safe = matrix.clone()
    if invalid.any():
        safe[invalid] = torch.eye(2, device=matrix.device, dtype=matrix.dtype)
    xy = torch.linalg.solve(safe, right.unsqueeze(-1)).squeeze(-1)
    return torch.cat((xy, z.unsqueeze(-1)), dim=-1), ~invalid


def _matched(targets, indices, field: str, device, dtype=None) -> Tensor:
    parts = []
    for target, (_, target_index) in zip(targets, indices):
        index = torch.as_tensor(target_index, dtype=torch.int64, device=device)
        value = target[field].to(device=device)
        value = value.index_select(0, index)
        if dtype is not None:
            value = value.to(dtype=dtype)
        parts.append(value)
    return torch.cat(parts, dim=0)


def _matched_batch_indices(indices, device) -> Tensor:
    return torch.cat(tuple(torch.full(
        (len(target_index),), batch_index, dtype=torch.int64, device=device)
        for batch_index, (_, target_index) in enumerate(indices)), dim=0)


def _rectangle_corners(center_xz: Tensor, width: Tensor,
                       length: Tensor, yaw: Tensor) -> Tensor:
    signs = center_xz.new_tensor(((-1., -1.), (-1., 1.),
                                  (1., 1.), (1., -1.)))
    local = signs * torch.stack((length, width), dim=-1).unsqueeze(-2) * .5
    cosine, sine = yaw.cos(), yaw.sin()
    x = local[..., 0] * cosine.unsqueeze(-1) + local[..., 1] * sine.unsqueeze(-1)
    z = -local[..., 0] * sine.unsqueeze(-1) + local[..., 1] * cosine.unsqueeze(-1)
    return torch.stack((x, z), dim=-1) + center_xz.unsqueeze(-2)


def _inside(points: Tensor, center_xz: Tensor, width: Tensor,
            length: Tensor, yaw: Tensor) -> Tensor:
    relative = points - center_xz.unsqueeze(-2)
    cosine, sine = yaw.cos().unsqueeze(-1), yaw.sin().unsqueeze(-1)
    local_x = relative[..., 0] * cosine - relative[..., 1] * sine
    local_z = relative[..., 0] * sine + relative[..., 1] * cosine
    epsilon = 32 * torch.finfo(points.dtype).eps
    return ((local_x.abs() <= length.unsqueeze(-1) * .5 + epsilon)
            & (local_z.abs() <= width.unsqueeze(-1) * .5 + epsilon))


def _inside_precomputed(
        points: Tensor, center_xz: Tensor, half_width: Tensor,
        half_length: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    relative = points - center_xz.unsqueeze(-2)
    local_x = (relative[..., 0] * cosine.unsqueeze(-1)
               - relative[..., 1] * sine.unsqueeze(-1))
    local_z = (relative[..., 0] * sine.unsqueeze(-1)
               + relative[..., 1] * cosine.unsqueeze(-1))
    epsilon = 32 * torch.finfo(points.dtype).eps
    return ((local_x.abs() <= half_length.unsqueeze(-1) + epsilon)
            & (local_z.abs() <= half_width.unsqueeze(-1) + epsilon))


def _cross(first: Tensor, second: Tensor) -> Tensor:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _intersection_area_from_corners(
        first: Tensor, second: Tensor, first_valid: Tensor,
        second_valid: Tensor) -> Tensor:
    first_start, first_end = first, first.roll(-1, dims=-2)
    second_start, second_end = second, second.roll(-1, dims=-2)
    p = first_start.unsqueeze(-2)
    r = (first_end - first_start).unsqueeze(-2)
    q = second_start.unsqueeze(-3)
    s = (second_end - second_start).unsqueeze(-3)
    denominator = _cross(r, s)
    qp = q - p
    epsilon = 64 * torch.finfo(first.dtype).eps
    safe = torch.where(denominator.abs() > epsilon, denominator,
                       torch.ones_like(denominator))
    t = _cross(qp, s) / safe
    u = _cross(qp, r) / safe
    intersection_valid = ((denominator.abs() > epsilon)
                          & (t >= -epsilon) & (t <= 1 + epsilon)
                          & (u >= -epsilon) & (u <= 1 + epsilon))
    intersections = p + t.unsqueeze(-1) * r
    candidates = torch.cat((
        first, second, intersections.flatten(-3, -2)), dim=-2)
    valid = torch.cat((
        first_valid, second_valid, intersection_valid.flatten(-2, -1)), dim=-1)

    count = valid.sum(-1)
    centroid = (candidates * valid.unsqueeze(-1)).sum(-2) / count.clamp_min(
        1).unsqueeze(-1)
    angles = torch.atan2(
        candidates[..., 1] - centroid[..., 1:2],
        candidates[..., 0] - centroid[..., 0:1])
    angles = torch.where(valid, angles, torch.full_like(angles, 10.0))
    order = angles.argsort(-1)
    ordered = candidates.gather(
        -2, order.unsqueeze(-1).expand(*order.shape, 2))
    positions = torch.arange(
        ordered.shape[-2], device=ordered.device).reshape(
            *((1,) * (ordered.ndim - 2)), -1)
    edge_valid = positions < (count - 1).unsqueeze(-1)
    edge_sum = torch.where(
        edge_valid,
        _cross(ordered, ordered.roll(-1, dims=-2)),
        torch.zeros_like(angles)).sum(-1)
    first_point = ordered[..., 0, :]
    last_index = (count - 1).clamp_min(0)
    last_point = ordered.gather(
        -2, last_index[..., None, None].expand(*last_index.shape, 1, 2)
    ).squeeze(-2)
    area = .5 * (edge_sum + _cross(last_point, first_point)).abs()
    return torch.where(count >= 3, area, torch.zeros_like(area))


def paired_rotated_intersection_area(
        first_center: Tensor, first_width: Tensor, first_length: Tensor,
        first_yaw: Tensor, second_center: Tensor, second_width: Tensor,
        second_length: Tensor, second_yaw: Tensor) -> Tensor:
    """Vectorized exact candidate-vertex area for paired rotated rectangles."""
    first = _rectangle_corners(
        first_center, first_width, first_length, first_yaw)
    second = _rectangle_corners(
        second_center, second_width, second_length, second_yaw)
    first_valid = _inside(
        first, second_center, second_width, second_length, second_yaw)
    second_valid = _inside(
        second, first_center, first_width, first_length, first_yaw)
    return _intersection_area_from_corners(
        first, second, first_valid, second_valid)


def paired_iou3d(first_center: Tensor, first_dimensions: Tensor,
                 first_yaw: Tensor, second_center: Tensor,
                 second_dimensions: Tensor, second_yaw: Tensor) -> Tensor:
    first_h, first_w, first_l = first_dimensions.unbind(-1)
    second_h, second_w, second_l = second_dimensions.unbind(-1)
    bev = paired_rotated_intersection_area(
        first_center[..., (0, 2)], first_w, first_l, first_yaw,
        second_center[..., (0, 2)], second_w, second_l, second_yaw)
    first_low, first_high = first_center[..., 1] - first_h * .5, first_center[..., 1] + first_h * .5
    second_low, second_high = second_center[..., 1] - second_h * .5, second_center[..., 1] + second_h * .5
    vertical = (torch.minimum(first_high, second_high)
                - torch.maximum(first_low, second_low)).clamp_min(0)
    intersection = bev * vertical
    first_volume = first_h * first_w * first_l
    second_volume = second_h * second_w * second_l
    union = first_volume + second_volume - intersection
    return intersection / union.clamp_min(torch.finfo(union.dtype).eps)


def _decode_alpha(angle: Tensor) -> Tensor:
    bins = angle[..., :12].argmax(-1)
    residual = angle[..., 12:].gather(-1, bins.unsqueeze(-1)).squeeze(-1)
    alpha = bins.to(angle.dtype) * (2.0 * math.pi / 12.0) + residual
    return torch.remainder(alpha + math.pi, 2.0 * math.pi) - math.pi


def _decode_yaw_like_public_decoder(
        pred_alpha: Tensor, boxes: Tensor, image_widths: Tensor,
        heading_calibs: Tensor) -> Tensor:
    """Torch equivalent of decode_helper.decode_detections' alpha2ry."""
    pred_bbox_center_x = (
        boxes[..., 0] + .5 * (boxes[..., 3] - boxes[..., 2])
    ) * image_widths
    return torch.remainder(
        pred_alpha + torch.atan2(
            pred_bbox_center_x - heading_calibs[..., 0, 2],
            heading_calibs[..., 0, 0])
        + math.pi, 2.0 * math.pi) - math.pi


@torch.no_grad()
def matched_iou_depth_intervals(
        outputs, targets, indices, iou_threshold: float,
        decode_mean_sizes: Sequence[Sequence[float]],
        depth_min_m: float = 2.0, depth_max_m: float = 65.0,
        bisection_steps: int = 22,
        eligible_mask: Tensor | None = None,
        compact_supported_bisection: bool = True,
        fuse_bidirectional_bisection: bool = True,
        reuse_static_iou_geometry: bool = True):
    device = outputs['pred_depth'].device
    dtype = outputs['pred_depth'].dtype
    source_batch = _matched_batch_indices(indices, device)
    source_query = torch.cat(tuple(torch.as_tensor(
        source_index, dtype=torch.int64, device=device)
        for source_index, _ in indices), dim=0)
    pair = (source_batch, source_query)
    boxes = outputs['pred_boxes'][pair].detach()
    logits = outputs['pred_logits'][pair].detach()
    dim_residual = outputs['pred_3d_dim'][pair].detach()
    angles = outputs['pred_angle'][pair].detach()

    decode_means = torch.as_tensor(
        decode_mean_sizes, device=device, dtype=dtype)
    predicted_labels = logits.argmax(-1)
    pred_dimensions = (
        dim_residual + decode_means[predicted_labels]).clamp_min(.05)
    centers_uv = boxes[..., :2] * torch.stack(tuple(
        target['projective_input_size'].to(device=device, dtype=dtype)
        for target in targets), dim=0)[source_batch]
    calibs_per_image = torch.stack(tuple(
        target['projective_image_effective_calib'].to(
            device=device, dtype=dtype) for target in targets), dim=0)
    calibs = calibs_per_image[source_batch]

    target_depth_virtual = _matched(
        targets, indices, 'depth', device, dtype).reshape(-1)
    depth_scale = _matched(
        targets, indices, 'depth_unit_scale', device, dtype).reshape(-1)
    gt_depth_m = target_depth_virtual / depth_scale.clamp_min(1e-6)
    gt_dimensions = _matched(
        targets, indices, 'src_size_3d', device, dtype)
    gt_yaw = _matched(
        targets, indices, 'projective_rotation_y', device, dtype).reshape(-1)
    gt_uv_norm = _matched(
        targets, indices, 'boxes_3d', device, dtype)[..., :2]
    input_sizes = torch.stack(tuple(
        target['projective_input_size'].to(device=device, dtype=dtype)
        for target in targets), dim=0)[source_batch]
    gt_uv = gt_uv_norm * input_sizes

    def centers_at(uv: Tensor, depth_m: Tensor) -> tuple[Tensor, Tensor]:
        return _centers_on_projected_rays(
            uv, depth_m.unsqueeze(-1), calibs)

    gt_center, gt_ray_valid = centers_at(gt_uv, gt_depth_m)
    _, pred_ray_valid = centers_at(centers_uv, gt_depth_m)
    pred_alpha = _decode_alpha(angles)
    # Match the detector's public decoder exactly: its supervised alpha is
    # defined against the predicted 2-D box midpoint, not the separately
    # predicted projected 3-D centre.  Decode yaw once and keep it fixed while
    # depth alone moves along the projected 3-D-centre ray.
    decoder_image_widths = torch.stack(tuple(
        target['img_size'].to(device=device, dtype=dtype)
        for target in targets), dim=0)[source_batch, 0]
    heading_calibs = _matched(targets, indices, 'calibs', device, dtype)
    pred_yaw = _decode_yaw_like_public_decoder(
        pred_alpha, boxes, decoder_image_widths, heading_calibs)

    pred_h, pred_w, pred_l = pred_dimensions.unbind(-1)
    gt_h, gt_w, gt_l = gt_dimensions.unbind(-1)
    pred_cosine, pred_sine = pred_yaw.cos().float(), pred_yaw.sin().float()
    gt_cosine, gt_sine = gt_yaw.cos().float(), gt_yaw.sin().float()
    pred_half_width = (pred_w * .5).float()
    pred_half_length = (pred_l * .5).float()
    gt_half_width = (gt_w * .5).float()
    gt_half_length = (gt_l * .5).float()
    gt_center_xz = gt_center[..., (0, 2)].float()
    pred_corner_offsets = _rectangle_corners(
        torch.zeros_like(gt_center[..., (0, 2)]),
        pred_w, pred_l, pred_yaw).float()
    gt_corners = _rectangle_corners(
        gt_center[..., (0, 2)], gt_w, gt_l, gt_yaw).float()
    pred_volume = (pred_h * pred_w * pred_l).float()
    gt_volume = (gt_h * gt_w * gt_l).float()
    gt_low = (gt_center[..., 1] - gt_h * .5).float()
    gt_high = (gt_center[..., 1] + gt_h * .5).float()

    finite = (torch.isfinite(pred_dimensions).all(-1)
              & torch.isfinite(gt_dimensions).all(-1)
              & torch.isfinite(gt_depth_m)
              & (gt_depth_m > depth_min_m) & (gt_depth_m < depth_max_m)
              & gt_ray_valid & pred_ray_valid)

    if eligible_mask is None:
        eligible_mask = torch.ones_like(finite)
    else:
        eligible_mask = eligible_mask.to(device=device, dtype=torch.bool)
        if eligible_mask.shape != finite.shape:
            raise ValueError("eligible interval mask must have shape [N]")

    def iou_at(depth_m: Tensor, row_index: Tensor) -> Tensor:
        center, ray_valid = _centers_on_projected_rays(
            centers_uv.index_select(0, row_index), depth_m.unsqueeze(-1),
            calibs.index_select(0, row_index))
        if reuse_static_iou_geometry:
            center = center.float()
            row_pred_h = pred_h.index_select(0, row_index).float()
            first_center_xz = center[..., (0, 2)]
            first = (pred_corner_offsets.index_select(
                0, row_index) + first_center_xz.unsqueeze(-2))
            second = gt_corners.index_select(0, row_index)
            first_valid = _inside_precomputed(
                first,
                gt_center_xz.index_select(0, row_index),
                gt_half_width.index_select(0, row_index),
                gt_half_length.index_select(0, row_index),
                gt_cosine.index_select(0, row_index),
                gt_sine.index_select(0, row_index))
            second_valid = _inside_precomputed(
                second, first_center_xz,
                pred_half_width.index_select(0, row_index),
                pred_half_length.index_select(0, row_index),
                pred_cosine.index_select(0, row_index),
                pred_sine.index_select(0, row_index))
            bev = _intersection_area_from_corners(
                first, second, first_valid, second_valid)
            first_low = center[..., 1] - row_pred_h * .5
            first_high = center[..., 1] + row_pred_h * .5
            vertical = (torch.minimum(
                first_high, gt_high.index_select(0, row_index))
                - torch.maximum(
                    first_low, gt_low.index_select(0, row_index))
            ).clamp_min(0)
            intersection = bev * vertical
            union = (pred_volume.index_select(0, row_index)
                     + gt_volume.index_select(0, row_index)
                     - intersection)
            iou = (intersection / union.clamp_min(
                torch.finfo(union.dtype).eps)).to(dtype)
        else:
            iou = paired_iou3d(
                center.float(),
                pred_dimensions.index_select(0, row_index).float(),
                pred_yaw.index_select(0, row_index).float(),
                gt_center.index_select(0, row_index).float(),
                gt_dimensions.index_select(0, row_index).float(),
                gt_yaw.index_select(0, row_index).float()).to(dtype)
        return torch.where(ray_valid, iou, torch.zeros_like(iou))

    candidate_index = torch.nonzero(
        finite & eligible_mask, as_tuple=False).reshape(-1)
    iou_gt = torch.zeros_like(gt_depth_m)
    if candidate_index.numel():
        candidate_iou = iou_at(
            gt_depth_m.index_select(0, candidate_index), candidate_index)
        iou_gt.index_copy_(0, candidate_index, candidate_iou)
    supported = finite & eligible_mask & (iou_gt >= float(iou_threshold))

    # Unsupported rows can never contribute to the interval loss.  Compact
    # them away before the 44 expensive rotated-IoU bisection evaluations.
    supported_index = torch.nonzero(supported, as_tuple=False).reshape(-1)
    bisection_index = (
        supported_index if compact_supported_bisection else candidate_index)
    left = gt_depth_m.clone()
    right = gt_depth_m.clone()
    if bisection_index.numel():
        gt_supported = gt_depth_m.index_select(0, bisection_index)
        lower = torch.full_like(gt_supported, float(depth_min_m))
        upper = torch.full_like(gt_supported, float(depth_max_m))
        if fuse_bidirectional_bisection:
            doubled_index = torch.cat((bisection_index, bisection_index))
            boundary_iou = iou_at(
                torch.cat((lower, upper)), doubled_index)
            lower_iou, upper_iou = boundary_iou.chunk(2)
            left_outside, left_inside = lower, gt_supported
            right_inside, right_outside = gt_supported, upper
            for _ in range(int(bisection_steps)):
                left_midpoint = (left_outside + left_inside) * .5
                right_midpoint = (right_inside + right_outside) * .5
                midpoint_inside = iou_at(
                    torch.cat((left_midpoint, right_midpoint)),
                    doubled_index).ge(float(iou_threshold))
                left_midpoint_inside, right_midpoint_inside = (
                    midpoint_inside.chunk(2))
                left_inside = torch.where(
                    left_midpoint_inside, left_midpoint, left_inside)
                left_outside = torch.where(
                    left_midpoint_inside, left_outside, left_midpoint)
                right_inside = torch.where(
                    right_midpoint_inside, right_midpoint, right_inside)
                right_outside = torch.where(
                    right_midpoint_inside, right_outside, right_midpoint)
        else:
            lower_iou = iou_at(lower, bisection_index)
            upper_iou = iou_at(upper, bisection_index)
            left_outside, left_inside = lower, gt_supported
            for _ in range(int(bisection_steps)):
                midpoint = (left_outside + left_inside) * .5
                midpoint_inside = iou_at(
                    midpoint, bisection_index).ge(float(iou_threshold))
                left_inside = torch.where(
                    midpoint_inside, midpoint, left_inside)
                left_outside = torch.where(
                    midpoint_inside, left_outside, midpoint)
            right_inside, right_outside = gt_supported, upper
            for _ in range(int(bisection_steps)):
                midpoint = (right_inside + right_outside) * .5
                midpoint_inside = iou_at(
                    midpoint, bisection_index).ge(float(iou_threshold))
                right_inside = torch.where(
                    midpoint_inside, midpoint, right_inside)
                right_outside = torch.where(
                    midpoint_inside, right_outside, midpoint)
        compact_left = torch.where(
            lower_iou >= float(iou_threshold), lower, left_inside)
        compact_right = torch.where(
            upper_iou >= float(iou_threshold), upper, right_inside)
        left.index_copy_(0, bisection_index, compact_left)
        right.index_copy_(0, bisection_index, compact_right)
    nondegenerate = (gt_depth_m > left) & (right > gt_depth_m)
    valid = supported & nondegenerate
    return {
        'left_virtual': left * depth_scale,
        'right_virtual': right * depth_scale,
        'gt_virtual': target_depth_virtual,
        'valid': valid,
        'supported': supported,
        'iou_at_gt': iou_gt,
        'predicted_labels': predicted_labels,
    }


def asymmetric_interval_and_uncertainty_loss(
        outputs, targets, indices, num_boxes, car_class_id: int,
        iou_threshold: float,
        decode_mean_sizes: Sequence[Sequence[float]],
        matched_prediction: Tensor | None = None,
        matched_residual: Tensor | None = None,
        compact_supported_bisection: bool = True,
        fuse_bidirectional_bisection: bool = True,
        reuse_static_iou_geometry: bool = True,
        precomputed_interval: dict[str, Tensor] | None = None):
    device = outputs['pred_depth'].device
    dtype = outputs['pred_depth'].dtype
    source_batch = _matched_batch_indices(indices, device)
    source_query = torch.cat(tuple(torch.as_tensor(
        source_index, dtype=torch.int64, device=device)
        for source_index, _ in indices), dim=0)
    predicted = (
        outputs['pred_depth'][source_batch, source_query]
        if matched_prediction is None else matched_prediction)
    if predicted.shape != (source_query.numel(), 2):
        raise ValueError("matched depth prediction must have shape [N,2]")
    mean, log_scale = predicted[:, 0], predicted[:, 1]
    labels = _matched(targets, indices, 'labels', device).reshape(-1).long()
    interval = precomputed_interval
    if interval is None:
        interval = matched_iou_depth_intervals(
            outputs, targets, indices, iou_threshold, decode_mean_sizes,
            eligible_mask=labels.eq(int(car_class_id)),
            compact_supported_bisection=compact_supported_bisection,
            fuse_bidirectional_bisection=fuse_bidirectional_bisection,
            reuse_static_iou_geometry=reuse_static_iou_geometry)
    else:
        required = {
            'left_virtual', 'right_virtual', 'gt_virtual', 'valid',
            'supported', 'iou_at_gt', 'predicted_labels'}
        missing = sorted(required - set(interval))
        if missing:
            raise ValueError(
                f"precomputed interval is missing fields: {missing}")
        if interval['valid'].shape != labels.shape:
            raise ValueError(
                "precomputed interval object count disagrees with matching")
    car = labels.eq(int(car_class_id))
    unique_matched_car_count = labels.new_zeros(())
    for target, (_, target_index) in zip(targets, indices):
        unique_target_index = torch.unique(torch.as_tensor(
            target_index, dtype=torch.int64, device=device))
        unique_labels = target['labels'].to(device=device).index_select(
            0, unique_target_index).reshape(-1).long()
        unique_matched_car_count += unique_labels.eq(int(car_class_id)).sum()
    finite_prediction = (
        torch.isfinite(mean) & torch.isfinite(log_scale))
    valid = interval['valid'] & car & finite_prediction
    gt = interval['gt_virtual']
    finite_target = torch.isfinite(gt)
    safe_mean = torch.where(finite_prediction, mean, torch.zeros_like(mean))
    safe_gt = torch.where(finite_target, gt, torch.zeros_like(gt))
    error = (safe_mean - safe_gt).abs()
    left_width = (safe_gt - torch.where(
        valid, interval['left_virtual'], safe_gt)).clamp_min(0)
    right_width = (torch.where(
        valid, interval['right_virtual'], safe_gt) - safe_gt).clamp_min(0)
    side_width = torch.where(safe_mean <= safe_gt, left_width, right_width)
    side_width = side_width.clamp_min(torch.finfo(dtype).eps)
    inside = valid & (error <= side_width)
    outside = valid & ~inside

    # Asymmetric Huber: the GT is the mode, while the left/right feasible
    # widths independently set where the quadratic core changes to a linear
    # tail.  Its mean-depth gradient magnitude is min(|error|, side_width), so
    # a narrow feasible side cannot create the old 1/width gradient explosion.
    per_object_huber = torch.where(
        inside, .5 * error.square(),
        side_width * error - .5 * side_width.square())
    per_object_huber = torch.where(
        valid, per_object_huber, torch.zeros_like(per_object_huber))

    # If predicted geometry cannot form a valid feasible interval, preserve
    # the detector's native Laplace supervision.  This avoids silently
    # discarding roughly half of the matched objects.
    fallback = ~valid & finite_prediction & finite_target
    native_laplace = (
        LAPLACE_FACTOR * torch.exp(-log_scale.float()) * error.float()
        + log_scale.float())
    mean_loss_terms = per_object_huber.float() + torch.where(
        fallback, native_laplace, torch.zeros_like(native_laplace))
    interval_loss = mean_loss_terms.sum() / float(num_boxes)

    # Calibrate only the independent uncertainty output.  The detached error
    # makes this term exactly incapable of changing the predicted mean depth.
    detached_error = (safe_mean.detach() - safe_gt).abs()
    per_object_uncertainty = (
        LAPLACE_FACTOR * torch.exp(-log_scale.float())
        * detached_error.float()
        + log_scale.float())
    per_object_uncertainty = torch.where(
        valid, per_object_uncertainty,
        torch.zeros_like(per_object_uncertainty))
    # This objective exists only for objects with a valid interval.  Normalize
    # by that participating population, not by every matched object; otherwise
    # its apparent scale and effective weight grow merely because interval
    # coverage improves during training.
    valid_uncertainty_count = valid.sum().clamp_min(1).to(
        dtype=per_object_uncertainty.dtype)
    uncertainty_loss = (
        per_object_uncertainty.sum() / valid_uncertainty_count)
    if matched_residual is not None:
        residual_receipt = matched_residual
    else:
        conditioned_metadata = outputs.get(
            'geometry_conditioned_interval_depth', {})
        all_residuals = conditioned_metadata.get('residual')
        residual_receipt = (
            all_residuals[source_batch, source_query]
            if all_residuals is not None else mean.new_empty((0,)))
    receipt = {
        'matched_count': torch.as_tensor(mean.numel(), device=device),
        'unique_matched_car_count': unique_matched_car_count.detach(),
        'eligible_car_count': car.sum().detach(),
        'valid_interval_count': valid.sum().detach(),
        'supported_at_gt_count': interval['supported'].sum().detach(),
        'fallback_native_count': fallback.sum().detach(),
        'inside_count': inside.sum().detach(),
        'outside_count': outside.sum().detach(),
        'interval_loss': interval_loss.detach(),
        'uncertainty_loss': uncertainty_loss.detach(),
        'uncertainty_normalizer_count': valid.sum().detach(),
        'absolute_error_virtual': error[car & finite_prediction & finite_target].detach(),
        'supported_absolute_error_virtual': error[valid].detach(),
        'unsupported_absolute_error_virtual': error[car & ~valid & finite_prediction & finite_target].detach(),
        'inside_absolute_error_virtual': error[inside].detach(),
        'outside_absolute_error_virtual': error[outside].detach(),
        'outside_boundary_distance_virtual': torch.where(
            safe_mean < interval['left_virtual'],
            interval['left_virtual'] - safe_mean,
            safe_mean - interval['right_virtual'])[outside].detach(),
        'left_width_virtual': left_width[valid].detach(),
        'right_width_virtual': right_width[valid].detach(),
        'predicted_depth_virtual': mean.detach(),
        'predicted_residual_virtual': residual_receipt.detach(),
        'iou_at_gt': interval['iou_at_gt'][car].detach(),
    }
    return interval_loss, uncertainty_loss, receipt
