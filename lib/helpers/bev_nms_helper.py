"""Class-wise BEV NMS for decoded MonoDGP/KITTI predictions."""

import cv2
import numpy as np


def _bev_polygon(prediction):
    _, width, length = map(float, prediction[6:9])
    x, _, z = map(float, prediction[9:12])
    yaw = float(prediction[12])
    local = np.asarray([
        [length / 2, width / 2],
        [length / 2, -width / 2],
        [-length / 2, -width / 2],
        [-length / 2, width / 2],
    ], dtype=np.float32)
    rotation = np.asarray([
        [np.cos(yaw), np.sin(yaw)],
        [-np.sin(yaw), np.cos(yaw)],
    ], dtype=np.float32)
    return cv2.convexHull(
        local @ rotation.T + np.asarray([x, z], dtype=np.float32))


def _pairwise_bev_iou(predictions):
    count = len(predictions)
    overlaps = np.zeros((count, count), dtype=np.float32)
    if count == 0:
        return overlaps
    polygons = [_bev_polygon(prediction) for prediction in predictions]
    areas = np.asarray([
        max(0.0, float(prediction[7] * prediction[8]))
        for prediction in predictions
    ], dtype=np.float32)
    for first in range(count):
        for second in range(first + 1, count):
            if int(predictions[first][0]) != int(predictions[second][0]):
                continue
            intersection, _ = cv2.intersectConvexConvex(
                polygons[first], polygons[second])
            intersection = min(
                float(areas[first]), float(areas[second]),
                max(0.0, float(intersection)))
            union = float(areas[first] + areas[second] - intersection)
            iou = intersection / union if union > 0 else 0.0
            overlaps[first, second] = iou
            overlaps[second, first] = iou
    return overlaps


def _keep_indices(predictions, overlaps, threshold):
    order = sorted(
        range(len(predictions)),
        key=lambda index: float(predictions[index][13]),
        reverse=True)
    kept = []
    while order:
        current = order.pop(0)
        kept.append(current)
        order = [
            candidate for candidate in order
            if (int(predictions[current][0]) != int(predictions[candidate][0])
                or overlaps[current, candidate] <= threshold)
        ]
    return kept


def classwise_bev_nms_variants(results, thresholds):
    """Compute several thresholds while reusing each image's BEV IoU matrix."""
    normalized = tuple(float(threshold) for threshold in thresholds)
    if not normalized:
        return {}
    if len(set(normalized)) != len(normalized):
        raise ValueError('BEV NMS thresholds must be unique')
    if any(not 0 <= threshold <= 1 for threshold in normalized):
        raise ValueError('BEV NMS thresholds must be in [0, 1]')

    variants = {threshold: {} for threshold in normalized}
    for image_id, predictions in results.items():
        overlaps = _pairwise_bev_iou(predictions)
        for threshold in normalized:
            kept = _keep_indices(predictions, overlaps, threshold)
            variants[threshold][image_id] = [
                predictions[index] for index in kept
            ]
    return variants
