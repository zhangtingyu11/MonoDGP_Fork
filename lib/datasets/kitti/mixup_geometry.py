"""CPU-only camera-normalized MixUp geometry and image composition."""

import cv2
import numpy as np


def camera_normalized_mixup_geometry(recipient_p2, donor_p2):
    """Map donor pixels and camera coordinates into the recipient camera.

    Writing each projection as ``P = [M | t]``, the pixel homography
    ``H = M_r M_d^-1`` and coordinate translation
    ``delta = M_d^-1 t_d - M_r^-1 t_r`` satisfy, for every 3-D point X,

        P_r [X + delta, 1] = H P_d [X, 1].

    The translation preserves physical box dimensions and yaw while making
    warped donor pixels, transformed annotations and the recipient P2 exact.
    """
    recipient_p2 = np.asarray(recipient_p2, dtype=np.float64)
    donor_p2 = np.asarray(donor_p2, dtype=np.float64)
    if recipient_p2.shape != (3, 4) or donor_p2.shape != (3, 4):
        raise ValueError('cross-focal MixUp requires 3x4 projection matrices')
    recipient_m = recipient_p2[:, :3]
    donor_m = donor_p2[:, :3]
    try:
        donor_m_inverse = np.linalg.inv(donor_m)
        recipient_m_inverse = np.linalg.inv(recipient_m)
    except np.linalg.LinAlgError as error:
        raise ValueError('cross-focal MixUp received singular P2') from error
    homography = recipient_m @ donor_m_inverse
    translation = (
        donor_m_inverse @ donor_p2[:, 3]
        - recipient_m_inverse @ recipient_p2[:, 3])
    if not (np.isfinite(homography).all()
            and np.isfinite(translation).all()):
        raise ValueError('cross-focal MixUp geometry is not finite')
    return homography, translation


def transform_projective_box(box, homography):
    """Transform all four corners and return their axis-aligned envelope."""
    box = np.asarray(box, dtype=np.float64)
    corners = np.array([
        [box[0], box[1], 1.0],
        [box[2], box[1], 1.0],
        [box[0], box[3], 1.0],
        [box[2], box[3], 1.0],
    ], dtype=np.float64)
    transformed = corners @ np.asarray(homography, dtype=np.float64).T
    denominator = transformed[:, 2:3]
    if np.any(np.abs(denominator) <= np.finfo(np.float64).eps):
        raise ValueError('cross-focal MixUp box maps to projective infinity')
    transformed = transformed[:, :2] / denominator
    return np.array([
        transformed[:, 0].min(), transformed[:, 1].min(),
        transformed[:, 0].max(), transformed[:, 1].max(),
    ], dtype=np.float32)


def warp_and_blend_mixup(primary, donor, donor_to_input, output_size,
                         valid_threshold=0.999):
    """Warp and blend only fully supported donor pixels.

    The support image is warped with the same bilinear sampler as RGB. Pixels
    below the strict threshold retain the primary image, so border fill never
    darkens the training input.
    """
    width, height = map(int, output_size)
    primary = np.asarray(primary, dtype=np.uint8)
    donor = np.asarray(donor, dtype=np.uint8)
    matrix = np.asarray(donor_to_input, dtype=np.float64)
    warped = cv2.warpPerspective(
        donor, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    support = cv2.warpPerspective(
        np.ones(donor.shape[:2], dtype=np.float32), matrix,
        (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = support > float(valid_threshold)
    blended = primary.copy()
    if np.any(valid):
        blended[valid] = (
            (primary[valid].astype(np.uint16)
             + warped[valid].astype(np.uint16)) // 2).astype(np.uint8)
    return blended, valid, support


def merge_mixup_object_regions(primary_region, donor_region,
                               donor_valid_mask):
    """Merge donor object regions only where donor pixels were blended."""
    primary_region = np.asarray(primary_region, dtype=bool)
    donor_region = np.asarray(donor_region, dtype=bool)
    donor_valid_mask = np.asarray(donor_valid_mask, dtype=bool)
    if not (primary_region.shape == donor_region.shape
            == donor_valid_mask.shape):
        raise ValueError('MixUp region masks must have identical shapes')
    return primary_region | (donor_region & donor_valid_mask)


def mixup_box_valid_ratio(box, donor_valid_mask):
    """Return the visible box fraction backed by fully valid donor pixels."""
    donor_valid_mask = np.asarray(donor_valid_mask, dtype=bool)
    if donor_valid_mask.ndim != 2:
        raise ValueError('MixUp donor valid mask must be two-dimensional')
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError('MixUp box must be a finite xyxy vector')
    height, width = donor_valid_mask.shape
    ymin = int(np.clip(box[1], 0, height))
    ymax = int(np.clip(box[3], 0, height))
    xmin = int(np.clip(box[0], 0, width))
    xmax = int(np.clip(box[2], 0, width))
    if ymax <= ymin or xmax <= xmin:
        return 0.0
    return float(donor_valid_mask[ymin:ymax, xmin:xmax].mean())
