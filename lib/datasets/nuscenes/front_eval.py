"""nuScenes metric definitions on a declared CAM_FRONT/Car subset.

This is not the official 360-degree ten-class benchmark. The optional NDS-like
summary explicitly assigns zero velocity/attribute contributions; no official
metric object or installed devkit code is modified.
"""
import numpy as np
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from .front_geometry import transform_points, projected_box_intersection


CAMERA_BOX_BASIS = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def decoded_to_world(row, record):
    """Existing detector decoded row: class, alpha, bbox4, hwl3, bottom3, yaw, score."""
    row = np.asarray(row, dtype=np.float64)
    if row.shape != (14,) or not np.isfinite(row).all() or np.any(row[6:9] <= 0):
        raise ValueError('Invalid/non-finite decoded detection; refusing silent filtering')
    h, w, length = row[6:9]
    centre = row[9:12].copy()
    centre[1] -= h / 2
    yaw = row[12]
    rotation = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                         [-np.sin(yaw), 0, np.cos(yaw)]]) @ CAMERA_BOX_BASIS
    transform = np.asarray(record['world_from_camera'])
    return Box(transform_points(centre[None], transform)[0], [w, length, h],
               Quaternion(matrix=transform[:3, :3] @ rotation), score=float(row[13]))


def in_front_scope(box, record):
    if np.linalg.norm(box.center[:2] - np.asarray(record['ego_translation'])[:2]) >= 50:
        return False
    corners = transform_points(box.corners().T, np.asarray(record['camera_from_world']))
    return projected_box_intersection(corners, record['intrinsic'], record['image_size']) is not None


def as_detection(box, sample_token, num_pts=-1):
    return DetectionBox(sample_token=sample_token, translation=tuple(box.center),
        size=tuple(box.wlh), rotation=tuple(box.orientation.elements), velocity=(0., 0.),
        num_pts=num_pts, detection_name='car', detection_score=float(box.score), attribute_name='')


def ground_truth(records):
    result = EvalBoxes()
    for record in records:
        boxes = []
        for ann in record['annotations']:
            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']), score=1.)
            if not in_front_scope(box, record) or ann['num_pts'] <= 0:
                raise ValueError('Native cache violates declared GT filtering protocol')
            boxes.append(as_detection(box, record['sample_token'], ann['num_pts']))
        result.add_boxes(record['sample_token'], boxes)
    return result


def predictions(results, records):
    expected_ids = {int(r['id']) for r in records}
    if set(map(int, results)) != expected_ids:
        raise ValueError('Predictions must include every evaluated image, including empty images')
    result = EvalBoxes()
    counts = dict(input_car=0, outside_scope=0, retained=0)
    for record in records:
        boxes = []
        for row in results[int(record['id'])]:
            if int(row[0]) != 1:  # Exp63's retained class ID for Car.
                continue
            counts['input_car'] += 1
            box = decoded_to_world(row, record)
            if not in_front_scope(box, record):
                counts['outside_scope'] += 1
                continue
            # Keep the original detection score, even if depth precision makes
            # it >1. Official accumulate only ranks scores; no sigmoid/clamp.
            boxes.append(as_detection(box, record['sample_token']))
        if len(boxes) > 500:
            raise ValueError('More than 500 detections in one sample')
        counts['retained'] += len(boxes)
        result.add_boxes(record['sample_token'], boxes)
    return result, counts


def evaluate_boxes(gt, pred):
    if set(gt.sample_tokens) != set(pred.sample_tokens):
        raise ValueError('GT and prediction sample sets must match')
    metrics, curves = {}, {}
    for distance in (.5, 1., 2., 4.):
        curve = accumulate(gt, pred, 'car', center_distance, distance)
        curves[distance] = curve
        metrics[f'front_car_AP_{distance:g}m'] = 100 * float(calc_ap(curve, .1, .1))
    mean_ap = float(np.mean([metrics[f'front_car_AP_{d:g}m'] for d in (.5, 1., 2., 4.)]))
    metrics['front_car_AP'] = mean_ap
    for name, key in [('ATE', 'trans_err'), ('ASE', 'scale_err'), ('AOE', 'orient_err')]:
        metrics[f'front_car_{name}'] = float(calc_tp(curves[2.], .1, key))
    quality = sum(max(0., 1-metrics[f'front_car_{name}']) for name in ('ATE', 'ASE', 'AOE'))
    metrics['front_car_custom_NDS_no_velocity_attribute'] = (5*mean_ap + 100*quality)/10
    return dict(selection_score=mean_ap, selection_metric='front_car_AP', metrics=metrics)
