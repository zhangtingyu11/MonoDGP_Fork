from pathlib import Path

import numpy as np
import torch

from lib.datasets.kitti.mixup_geometry import (
    camera_normalized_mixup_geometry,
    merge_mixup_object_regions,
    mixup_box_valid_ratio,
    transform_projective_box,
    warp_and_blend_mixup,
)
from lib.helpers.config_helper import load_config
from lib.helpers.trainer_helper import (
    add_mixup_counts,
    collect_mixup_counts,
    mixup_monitor_payload,
)


ROOT = Path(__file__).resolve().parents[1]


def _project(projection, points):
    homogeneous = np.concatenate(
        (points, np.ones((len(points), 1), dtype=points.dtype)), axis=1)
    projected = homogeneous @ projection.T
    return projected[:, :2] / projected[:, 2:3]


def test_camera_normalized_geometry_is_exact_for_full_p2():
    donor = np.array([
        [702.0, 0.8, 612.0, 43.0],
        [0.2, 699.0, 174.0, -0.4],
        [0.0003, -0.0002, 1.0, 0.006],
    ], dtype=np.float64)
    recipient = np.array([
        [735.0, -0.4, 606.0, 46.0],
        [0.1, 728.0, 181.0, -0.2],
        [-0.0001, 0.0003, 1.0, 0.004],
    ], dtype=np.float64)
    points = np.array([
        [-4.0, 1.2, 8.0],
        [0.3, -0.4, 21.0],
        [7.1, 1.8, 58.0],
    ], dtype=np.float64)

    homography, translation = camera_normalized_mixup_geometry(
        recipient, donor)
    donor_pixels = _project(donor, points)
    donor_pixels_h = np.concatenate(
        (donor_pixels, np.ones((len(points), 1))), axis=1)
    warped = donor_pixels_h @ homography.T
    warped = warped[:, :2] / warped[:, 2:3]
    recipient_pixels = _project(recipient, points + translation)

    np.testing.assert_allclose(warped, recipient_pixels, rtol=0, atol=2e-12)


def test_projective_box_uses_all_four_corners():
    box = np.array([10.0, 20.0, 50.0, 70.0])
    homography = np.array([
        [1.1, 0.04, 7.0],
        [-0.03, 0.9, 5.0],
        [0.0005, -0.0002, 1.0],
    ])
    transformed = transform_projective_box(box, homography)
    corners = np.array([
        [10.0, 20.0, 1.0], [50.0, 20.0, 1.0],
        [10.0, 70.0, 1.0], [50.0, 70.0, 1.0],
    ]) @ homography.T
    corners = corners[:, :2] / corners[:, 2:3]
    expected = np.array([
        corners[:, 0].min(), corners[:, 1].min(),
        corners[:, 0].max(), corners[:, 1].max(),
    ])

    np.testing.assert_allclose(transformed, expected, rtol=0, atol=4e-6)


def test_invalid_warp_border_never_darkens_primary_image():
    primary = np.full((12, 16, 3), 100, dtype=np.uint8)
    donor = np.full((8, 10, 3), 200, dtype=np.uint8)
    donor_to_input = np.array([
        [1.0, 0.0, 3.35],
        [0.0, 1.0, 2.65],
        [0.0, 0.0, 1.0],
    ])

    blended, valid, support = warp_and_blend_mixup(
        primary, donor, donor_to_input, (16, 12), valid_threshold=0.999)

    assert np.any(valid)
    assert np.any((support > 0.0) & ~valid)
    assert np.all(blended[valid] == 150)
    assert np.all(blended[~valid] == 100)


def test_donor_region_is_limited_to_pixels_that_were_actually_mixed():
    primary_region = np.zeros((5, 7), dtype=bool)
    primary_region[1:3, 1:3] = True
    donor_region = np.zeros_like(primary_region)
    donor_region[2:5, 3:7] = True
    donor_valid = np.zeros_like(primary_region)
    donor_valid[:4, :5] = True

    merged = merge_mixup_object_regions(
        primary_region, donor_region, donor_valid)

    expected = primary_region | (donor_region & donor_valid)
    np.testing.assert_array_equal(merged, expected)
    assert merged[3, 4]
    assert not merged[3, 5]
    assert not merged[4, 4]


def test_mixup_region_merge_rejects_shape_mismatch():
    with np.testing.assert_raises_regex(
            ValueError, 'identical shapes'):
        merge_mixup_object_regions(
            np.zeros((2, 3), dtype=bool),
            np.zeros((2, 3), dtype=bool),
            np.zeros((3, 2), dtype=bool))


def test_mixup_box_valid_ratio_distinguishes_outside_partial_and_full():
    valid = np.zeros((6, 8), dtype=bool)
    valid[:, 2:7] = True

    assert mixup_box_valid_ratio([0, 1, 2, 5], valid) == 0.0
    assert mixup_box_valid_ratio([1, 1, 5, 5], valid) == 0.75
    assert mixup_box_valid_ratio([2, 1, 7, 5], valid) == 1.0


def test_experiment31_changes_only_cross_focal_mixup_controls():
    exp30 = load_config(ROOT / 'configs/monodgp_exp30.yaml')
    exp31 = load_config(ROOT / 'configs/monodgp_exp31.yaml')

    assert exp30['dataset']['cross_focal_mixup'] is False
    assert exp31['dataset']['cross_focal_mixup'] is True
    assert exp31['dataset']['full_p2_projection'] is True
    assert exp31['dataset']['random_mixup3d'] == 0.5
    assert exp31['dataset']['mixup_valid_mask_threshold'] == 0.999
    assert exp31['dataset']['mixup_min_object_valid_ratio'] == 0.999
    assert exp31['dataset']['mixup_max_attempts'] == 50
    exp30_dataset = dict(exp30['dataset'])
    exp31_dataset = dict(exp31['dataset'])
    exp30_dataset.pop('cross_focal_mixup')
    exp31_dataset.pop('cross_focal_mixup')
    assert exp31_dataset == exp30_dataset
    for section in ('model', 'optimizer', 'lr_scheduler', 'tester'):
        assert exp31[section] == exp30[section]
    exp30_trainer = dict(exp30['trainer'])
    exp31_trainer = dict(exp31['trainer'])
    exp30_trainer.pop('swanlab')
    exp31_trainer.pop('swanlab')
    assert exp31_trainer == exp30_trainer


def test_mixup_monitor_reports_conditional_rates_and_coverage():
    targets = {
        'mixup_requested': torch.tensor([1.0, 1.0, 0.0, 1.0]),
        'mixup_applied': torch.tensor([1.0, 0.0, 0.0, 1.0]),
        'mixup_cross_focal': torch.tensor([1.0, 0.0, 0.0, 0.0]),
        'mixup_valid_ratio': torch.tensor([0.8, 0.0, 0.0, 1.0]),
        'mixup_attempts': torch.tensor([1.0, 50.0, 0.0, 2.0]),
        'mixup_reject_capacity': torch.tensor([0.0, 3.0, 0.0, 1.0]),
        'mixup_reject_geometry': torch.tensor([0.0, 1.0, 0.0, 0.0]),
        'mixup_reject_no_overlap': torch.tensor([0.0, 0.0, 0.0, 0.0]),
        'mixup_reject_partial_object': torch.tensor([0.0, 1.0, 0.0, 0.0]),
        'mixup_focal_scale_x': torch.tensor([1.1, 0.0, 0.0, 1.0]),
        'mixup_focal_scale_y': torch.tensor([1.2, 0.0, 0.0, 1.0]),
    }
    counts = collect_mixup_counts(targets)
    doubled = add_mixup_counts({}, counts)
    doubled = add_mixup_counts(doubled, counts)
    payload = mixup_monitor_payload(doubled, scope='test')

    assert payload['test跨焦距MixUp/请求样本比例'] == 0.75
    assert payload['test跨焦距MixUp/实际启用样本比例'] == 0.5
    assert np.isclose(payload['test跨焦距MixUp/请求后成功率'], 2 / 3)
    assert payload['test跨焦距MixUp/成功样本中跨P2比例'] == 0.5
    assert np.isclose(
        payload['test跨焦距MixUp/成功样本平均有效像素覆盖率'], 0.9)
    assert np.isclose(
        payload['test跨焦距MixUp/跨P2样本平均有效像素覆盖率'], 0.8)
    assert np.isclose(
        payload['test跨焦距MixUp/成功样本平均水平焦距倍率'], 1.05)
    assert np.isclose(
        payload['test跨焦距MixUp/跨P2样本平均水平焦距倍率'], 1.1)
    assert np.isclose(
        payload['test跨焦距MixUp/请求样本因供体目标部分可见取消比例'],
        1 / 3)
