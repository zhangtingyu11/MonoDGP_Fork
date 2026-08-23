"""CPU-only camera-normalized MixUp geometry and image composition."""

import cv2
import numpy as np


def mixup_object_requires_boundary_protection(class_type, write_list):
    """Whether a labelled object must not cross the MixUp support edge."""
    return class_type in write_list


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


def classify_box_canvas_visibility(box, output_size):
    """Classify an xyxy box as outside, partial, or complete on a canvas."""
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError('MixUp box must be a finite xyxy vector')
    width, height = map(float, output_size)
    if width <= 0.0 or height <= 0.0:
        raise ValueError('MixUp output size must be positive')
    if box[2] <= 0.0 or box[3] <= 0.0 or box[0] >= width or box[1] >= height:
        return 'outside'
    if box[0] < 0.0 or box[1] < 0.0 or box[2] > width or box[3] > height:
        return 'partial'
    return 'complete'


def warp_mixup_support(donor_shape, donor_to_input, output_size,
                       valid_threshold=0.999):
    """Return the exact donor support used by RGB composition."""
    height, width = map(int, donor_shape[:2])
    output_width, output_height = map(int, output_size)
    matrix = np.asarray(donor_to_input, dtype=np.float64)
    support = cv2.warpPerspective(
        np.ones((height, width), dtype=np.float32), matrix,
        (output_width, output_height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return support > float(valid_threshold), support


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
    valid, support = warp_mixup_support(
        donor.shape, matrix, output_size, valid_threshold)
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


def mixup_box_crosses_valid_boundary(box, donor_valid_mask,
                                     minimum_support_ratio=0.999):
    """Whether the donor-support edge cuts through the visible box area.

    A fully unsupported box stays purely primary and a fully supported box is
    mixed uniformly.  Only an intermediate ratio creates the forbidden
    within-object opacity seam.
    """
    ratio = mixup_box_valid_ratio(box, donor_valid_mask)
    return 0.0 < ratio < float(minimum_support_ratio), ratio


def mixup_box_full_support_ratio(box, donor_valid_mask):
    """Return support over the full transformed box, including image exterior.

    Unlike :func:`mixup_box_valid_ratio`, pixels outside the output canvas are
    counted as invalid.  This is the strict object-completeness contract used
    by Experiment 34: a box clipped by either the image boundary or warp
    support must not be treated as fully visible.
    """
    donor_valid_mask = np.asarray(donor_valid_mask, dtype=bool)
    if donor_valid_mask.ndim != 2:
        raise ValueError('MixUp donor valid mask must be two-dimensional')
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError('MixUp box must be a finite xyxy vector')

    xmin = int(np.floor(box[0]))
    ymin = int(np.floor(box[1]))
    xmax = int(np.ceil(box[2]))
    ymax = int(np.ceil(box[3]))
    full_width = xmax - xmin
    full_height = ymax - ymin
    if full_width <= 0 or full_height <= 0:
        return 0.0

    height, width = donor_valid_mask.shape
    clipped_xmin = int(np.clip(xmin, 0, width))
    clipped_xmax = int(np.clip(xmax, 0, width))
    clipped_ymin = int(np.clip(ymin, 0, height))
    clipped_ymax = int(np.clip(ymax, 0, height))
    if clipped_xmax <= clipped_xmin or clipped_ymax <= clipped_ymin:
        return 0.0
    supported = donor_valid_mask[
        clipped_ymin:clipped_ymax, clipped_xmin:clipped_xmax].sum()
    return float(supported / (full_width * full_height))


def projective_point(point, homography):
    """Apply a 3x3 projective transform to one finite image point."""
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError('MixUp point must be a finite xy vector')
    projected = np.asarray(homography, dtype=np.float64) @ np.array(
        [point[0], point[1], 1.0], dtype=np.float64)
    if abs(projected[2]) <= np.finfo(np.float64).eps:
        raise ValueError('cross-focal MixUp point maps to projective infinity')
    return (projected[:2] / projected[2]).astype(np.float32)


def classify_mixup_object_visibility(box, projected_center,
                                     donor_valid_mask,
                                     minimum_support_ratio=0.999):
    """Classify a donor target after the exact RGB warp.

    ``outside`` objects contribute no mixed pixels. ``partial`` objects would
    create visible pixels without a complete label. ``center_outside`` covers
    both an off-canvas 3-D center and a center outside the transformed 2-D
    annotation, which MonoDGP cannot encode when ``clip_2d`` is disabled.
    """
    ratio = mixup_box_full_support_ratio(box, donor_valid_mask)
    if ratio <= 0.0:
        return 'outside', ratio
    if ratio < float(minimum_support_ratio):
        return 'partial', ratio

    box = np.asarray(box, dtype=np.float64)
    center = np.asarray(projected_center, dtype=np.float64)
    height, width = donor_valid_mask.shape
    center_inside_image = (
        center.shape == (2,) and np.isfinite(center).all()
        and 0.0 <= center[0] < width and 0.0 <= center[1] < height)
    center_inside_box = (
        center_inside_image
        and box[0] <= center[0] <= box[2]
        and box[1] <= center[1] <= box[3])
    if not center_inside_box:
        return 'center_outside', ratio
    center_x = int(np.floor(center[0]))
    center_y = int(np.floor(center[1]))
    if not donor_valid_mask[center_y, center_x]:
        return 'center_outside', ratio
    return 'complete', ratio
