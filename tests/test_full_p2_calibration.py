import numpy as np
import torch

from lib.datasets.utils import angle2class
from lib.datasets.kitti.kitti_utils import Calibration
from lib.helpers.decode_helper import decode_detections
from lib.models.monodgp.iou3d_match_cost import pairwise_iou3d_match_cost


def _calibration(use_full_p2=True):
    return Calibration({
        'P2': np.array([
            [721.5, 0.35, 609.6, 44.8],
            [0.12, 721.2, 172.9, -0.31],
            [0.0002, -0.0001, 1.0, 0.0049],
        ], dtype=np.float64),
        'R0': np.eye(3, dtype=np.float64),
        'Tr_velo2cam': np.concatenate((
            np.eye(3, dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64)), axis=1),
    }, use_full_p2=use_full_p2)


def test_full_p2_projection_round_trip_uses_physical_z():
    calibration = _calibration()
    points = np.array([
        [-4.2, 1.1, 8.0],
        [0.3, -0.2, 21.0],
        [6.4, 1.8, 63.0],
    ], dtype=np.float64)

    image, depth = calibration.rect_to_img(points)
    restored = calibration.img_to_rect(image[:, 0], image[:, 1], depth)

    np.testing.assert_allclose(depth, points[:, 2], rtol=0, atol=0)
    np.testing.assert_allclose(restored, points, rtol=0, atol=2e-14)


def test_full_p2_physical_flip_preserves_projection_and_round_trip():
    calibration = _calibration()
    original_p2 = calibration.P2.copy()
    points = np.array([
        [-3.0, 0.8, 7.0],
        [1.7, -0.1, 18.0],
        [5.2, 1.4, 52.0],
    ], dtype=np.float64)
    width = 1242.0
    original_image, _ = calibration.rect_to_img(points)

    calibration.flip(np.array([width, 375.0]))
    reflected = points.copy()
    reflected[:, 0] *= -1
    flipped_image, depth = calibration.rect_to_img(reflected)

    np.testing.assert_allclose(
        flipped_image[:, 0], width - original_image[:, 0],
        rtol=0, atol=2e-13)
    np.testing.assert_allclose(
        flipped_image[:, 1], original_image[:, 1], rtol=0, atol=2e-13)
    np.testing.assert_allclose(
        calibration.img_to_rect(
            flipped_image[:, 0], flipped_image[:, 1], depth),
        reflected, rtol=0, atol=2e-14)

    calibration.flip(np.array([width, 375.0]))
    np.testing.assert_allclose(calibration.P2, original_p2, rtol=0, atol=0)


def test_full_p2_is_opt_in_and_legacy_projection_is_unchanged():
    calibration = _calibration(use_full_p2=False)
    point = np.array([[2.0, 1.0, 5.0]], dtype=np.float64)
    homogeneous = calibration.cart_to_hom(point) @ calibration.P2.T

    image, depth = calibration.rect_to_img(point)

    np.testing.assert_allclose(
        image, homogeneous[:, :2] / point[:, 2:3], rtol=0, atol=0)
    np.testing.assert_allclose(
        depth, homogeneous[:, 2] - calibration.P2[2, 3], rtol=0, atol=0)


def test_physical_ray_alpha_and_yaw_are_exact_under_flip():
    center = np.array([4.3, 1.2, 24.0], dtype=np.float64)
    yaw = 0.73
    alpha = np.remainder(
        yaw - np.arctan2(center[0], center[2]) + np.pi,
        2.0 * np.pi) - np.pi

    reflected = center.copy()
    reflected[0] *= -1
    reflected_yaw = np.remainder(
        np.pi - yaw + np.pi, 2.0 * np.pi) - np.pi
    reflected_alpha = np.remainder(
        reflected_yaw - np.arctan2(reflected[0], reflected[2]) + np.pi,
        2.0 * np.pi) - np.pi
    decoded_yaw = np.remainder(
        reflected_alpha + np.arctan2(reflected[0], reflected[2]) + np.pi,
        2.0 * np.pi) - np.pi

    np.testing.assert_allclose(
        reflected_alpha,
        np.remainder(np.pi - alpha + np.pi, 2.0 * np.pi) - np.pi,
        rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        decoded_yaw, reflected_yaw, rtol=0, atol=1e-15)


def test_full_p2_decoder_uses_reconstructed_3d_center_ray_for_yaw():
    calibration = _calibration()
    image_width, image_height = 1242.0, 375.0
    center = np.array([[5.0, 0.4, 28.0]], dtype=np.float64)
    projected, _ = calibration.rect_to_img(center)
    alpha = -0.31
    angle_class, angle_residual = angle2class(alpha)
    detection = np.zeros((1, 1, 37), dtype=np.float64)
    detection[0, 0, 0] = 1
    detection[0, 0, 1] = 0.9
    # Deliberately make the 2-D box midpoint disagree with the projected 3-D
    # centre; the corrected decoder must use the latter physical ray.
    detection[0, 0, 2:6] = [0.1, 0.5, 0.08, 0.2]
    detection[0, 0, 6] = center[0, 2]
    detection[0, 0, 7 + angle_class] = 10.0
    detection[0, 0, 19 + angle_class] = angle_residual
    detection[0, 0, 34] = projected[0, 0] / image_width
    detection[0, 0, 35] = projected[0, 1] / image_height
    detection[0, 0, 36] = 1.0

    decoded = decode_detections(
        detection,
        {'img_size': np.array([[image_width, image_height]]),
         'img_id': np.array([17])},
        [calibration], np.zeros((3, 3), dtype=np.float64), threshold=0.0)
    predicted = decoded[17][0]
    expected_yaw = np.remainder(
        alpha + np.arctan2(center[0, 0], center[0, 2]) + np.pi,
        2.0 * np.pi) - np.pi

    np.testing.assert_allclose(predicted[-2], expected_yaw, rtol=0, atol=1e-12)

    legacy = _calibration(use_full_p2=False)
    legacy_decoded = decode_detections(
        detection,
        {'img_size': np.array([[image_width, image_height]]),
         'img_id': np.array([17])},
        [legacy], np.zeros((3, 3), dtype=np.float64), threshold=0.0)
    legacy_expected = legacy.alpha2ry(alpha, detection[0, 0, 2] * image_width)
    np.testing.assert_allclose(
        legacy_decoded[17][0][-2], legacy_expected, rtol=0, atol=1e-12)


def test_full_p2_decoder_undoes_image_affine_before_raw_camera_decode():
    calibration = _calibration()
    input_size = np.array([1280.0, 384.0])
    affine = np.array([
        [1.07, 0.0, -31.0],
        [0.0, 1.07, 12.0],
        [0.0, 0.0, 1.0],
    ])
    affine_inverse = np.linalg.inv(affine)
    center = np.array([[3.2, 0.6, 24.0]], dtype=np.float64)
    projected, _ = calibration.rect_to_img(center)
    projected_input = affine @ np.r_[projected[0], 1.0]
    original_bbox = np.array([430.0, 105.0, 690.0, 315.0])
    top_left = affine @ np.r_[original_bbox[:2], 1.0]
    bottom_right = affine @ np.r_[original_bbox[2:], 1.0]
    bbox_input = np.r_[top_left[:2], bottom_right[:2]]

    detection = np.zeros((1, 1, 37), dtype=np.float64)
    detection[0, 0, 0:2] = [1.0, 0.9]
    detection[0, 0, 2:6] = [
        (bbox_input[0] + bbox_input[2]) / 2 / input_size[0],
        (bbox_input[1] + bbox_input[3]) / 2 / input_size[1],
        (bbox_input[2] - bbox_input[0]) / input_size[0],
        (bbox_input[3] - bbox_input[1]) / input_size[1],
    ]
    detection[0, 0, 6] = center[0, 2]
    detection[0, 0, 7] = 10.0
    detection[0, 0, 34:36] = (
        projected_input[:2] / input_size)
    detection[0, 0, 36] = 1.0

    decoded = decode_detections(
        detection,
        {
            'img_size': np.array([[1242.0, 375.0]]),
            'img_id': np.array([23]),
            'projective_input_size': input_size[None],
            'image_affine_inverse': affine_inverse[None],
        },
        [calibration], np.zeros((3, 3), dtype=np.float64), threshold=0.0)
    predicted = np.asarray(decoded[23][0])

    np.testing.assert_allclose(predicted[2:6], original_bbox, rtol=0, atol=2e-13)
    np.testing.assert_allclose(predicted[9:12], center[0], rtol=0, atol=2e-13)


def test_augmented_p2_and_input_box_height_recover_physical_depth():
    calibration = _calibration()
    physical_height = 1.52
    physical_depth = 27.0
    affine = np.array([
        [1.05, 0.0, -25.0],
        [0.0, 1.05, 8.0],
        [0.0, 0.0, 1.0],
    ])
    augmented_p2 = affine @ calibration.P2
    input_box_height = physical_height * augmented_p2[1, 1] / physical_depth

    recovered = physical_height * augmented_p2[1, 1] / input_box_height

    np.testing.assert_allclose(recovered, physical_depth, rtol=0, atol=1e-14)


def test_iou_matcher_uses_physical_projected_center_ray_for_yaw():
    dtype = torch.float64
    width, height = 1280.0, 384.0
    projection = torch.tensor([
        [720.0, 0.0, 610.0, 45.0],
        [0.0, 718.0, 175.0, -0.4],
        [0.0, 0.0, 1.0, 0.005],
    ], dtype=dtype)
    center = torch.tensor([5.0, 1.2, 25.0], dtype=dtype)
    homogeneous = torch.cat((center, center.new_ones(1)))
    projected = projection @ homogeneous
    uv = projected[:2] / projected[2]
    dimensions = torch.tensor([1.5, 1.7, 4.0], dtype=dtype)
    yaw = center.new_tensor(0.8)
    alpha = yaw - torch.atan2(center[0], center[2])
    angle_class, angle_residual = angle2class(float(alpha))
    angle = torch.zeros((1, 24), dtype=dtype)
    angle[0, angle_class] = 10.0
    angle[0, 12 + angle_class] = angle_residual
    # Asymmetric l/r makes the predicted 2-D box midpoint intentionally
    # different from the projected 3-D centre.  Exact heading must ignore it.
    box = torch.tensor([[
        uv[0] / width, uv[1] / height,
        0.18, 0.01, 0.05, 0.05,
    ]], dtype=dtype)
    outputs = {
        'pred_boxes': box,
        'pred_depth': torch.tensor([[center[2], 0.0]], dtype=dtype),
        'pred_3d_dim': dimensions[None],
        'pred_angle': angle,
    }
    target = {
        'labels': torch.tensor([1]),
        'boxes_3d': box.clone(),
        'depth': center[2:].clone(),
        'depth_unit_scale': torch.ones(1, dtype=dtype),
        'src_size_3d': dimensions[None],
        'projective_rotation_y': yaw.reshape(1, 1),
        'projective_input_size': torch.tensor([width, height], dtype=dtype),
        'projective_image_effective_calib': projection,
        'img_size': torch.tensor([width, height], dtype=dtype),
        'calibs': projection[None],
        'physical_ray_heading': torch.tensor(True),
    }

    iou, _ = pairwise_iou3d_match_cost(
        outputs, target, torch.zeros((3, 3), dtype=dtype))

    torch.testing.assert_close(iou, torch.ones_like(iou), rtol=0, atol=2e-12)
