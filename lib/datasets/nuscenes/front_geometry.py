"""Full-precision camera geometry shared by GT and prediction FOV filtering."""
import numpy as np
from shapely.geometry import MultiPoint, box as image_rectangle


NEAR_PLANE = 1e-3  # Numerical clipping plane in metres, not a centre-depth filter.
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def projected_box_intersection(corners_camera, intrinsic, image_size):
    """Return clipped projected silhouette bounds, or None for no positive area.

    Clip box edges in 3D before projecting; then intersect the projected convex
    hull with the image. Unlike corner-in-image tests, this retains boxes whose
    visible silhouette covers the image without any corner lying inside it.
    corners must follow nuScenes Box.corners() ordering.
    """
    corners = np.asarray(corners_camera, dtype=np.float64)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise ValueError('Expected eight finite camera-space corners')
    points = [p for p in corners if p[2] >= NEAR_PLANE]
    for a, b in EDGES:
        x, y = corners[a], corners[b]
        if (x[2] >= NEAR_PLANE) != (y[2] >= NEAR_PLANE):
            t = (NEAR_PLANE - x[2]) / (y[2] - x[2])
            points.append(x + t * (y - x))
    if len(points) < 3:
        return None
    projected = np.asarray(points) @ np.asarray(intrinsic, dtype=np.float64).T
    xy = projected[:, :2] / projected[:, 2:3]
    width, height = image_size
    if (xy[:, 0].max() <= 0 or xy[:, 1].max() <= 0
            or xy[:, 0].min() >= width or xy[:, 1].min() >= height):
        return None
    visible = MultiPoint(xy).convex_hull.intersection(image_rectangle(0, 0, width, height))
    if visible.is_empty or visible.area <= 0:
        return None
    return np.asarray(visible.bounds, dtype=np.float64)
