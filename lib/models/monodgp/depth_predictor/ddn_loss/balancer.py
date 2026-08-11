import torch
import torch.nn as nn
# based on
# https://github.com/TRAILab/CaDDN/blob/master/pcdet/models/backbones_3d/ffe/ddn_loss/balancer.py


class Balancer(nn.Module):
    def __init__(self, fg_weight, bg_weight, downsample_factor=1,
                 use_vectorized_rasterization=False):
        """
        Initialize fixed foreground/background loss balancer
        Args:
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.downsample_factor = downsample_factor
        self.use_vectorized_rasterization = bool(
            use_vectorized_rasterization)

    def forward(self, loss, gt_boxes2d, num_gt_per_img):
        """
        Forward pass
        Args:
            loss [torch.Tensor(B, H, W)]: Pixel-wise loss
            gt_boxes2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
        Returns:
            loss [torch.Tensor(1)]: Total loss after foreground/background balancing
            tb_dict [dict[float]]: All losses to log in tensorboard
        """
        # Compute masks
        if self.use_vectorized_rasterization:
            fg_mask = compute_fg_mask_vectorized(
                gt_boxes2d=gt_boxes2d,
                shape=loss.shape,
                num_gt_per_img=num_gt_per_img,
                downsample_factor=self.downsample_factor,
                device=loss.device)
        else:
            fg_mask = compute_fg_mask(gt_boxes2d=gt_boxes2d,
                                      shape=loss.shape,
                                      num_gt_per_img=num_gt_per_img,
                                      downsample_factor=self.downsample_factor,
                                      device=loss.device)
        bg_mask = ~fg_mask

        # Compute balancing weights
        weights = self.fg_weight * fg_mask + self.bg_weight * bg_mask
        num_pixels = fg_mask.sum() + bg_mask.sum()

        # Compute losses
        loss *= weights
        fg_loss = loss[fg_mask].sum() / num_pixels
        bg_loss = loss[bg_mask].sum() / num_pixels

        # Get total loss
        loss = fg_loss + bg_loss
        return loss


def compute_fg_mask(gt_boxes2d, shape, num_gt_per_img, downsample_factor=1, device=torch.device("cpu")):
    """
    Compute foreground mask for images
    Args:
        gt_boxes2d [torch.Tensor(B, N, 4)]: 2D box labels
        shape [torch.Size or tuple]: Foreground mask desired shape
        downsample_factor [int]: Downsample factor for image
        device [torch.device]: Foreground mask desired device
    Returns:
        fg_mask [torch.Tensor(shape)]: Foreground mask
    """
    #ipdb.set_trace()
    fg_mask = torch.zeros(shape, dtype=torch.bool, device=device)

    # Set box corners
    gt_boxes2d /= downsample_factor
    gt_boxes2d[:, :2] = torch.floor(gt_boxes2d[:, :2])
    gt_boxes2d[:, 2:] = torch.ceil(gt_boxes2d[:, 2:])
    gt_boxes2d = gt_boxes2d.long()
    height, width = int(shape[-2]), int(shape[-1])
    gt_boxes2d[:, 0::2].clamp_(0, width)
    gt_boxes2d[:, 1::2].clamp_(0, height)

    # Set all values within each box to True
    gt_boxes2d = gt_boxes2d.split(num_gt_per_img, dim=0)
    B = len(gt_boxes2d)
    for b in range(B):
        for n in range(gt_boxes2d[b].shape[0]):
            u1, v1, u2, v2 = gt_boxes2d[b][n]
            fg_mask[b, v1:v2, u1:u2] = True

    return fg_mask


def _validate_batch_layout(values, shape, num_gt_per_img):
    batch_size = int(shape[0])
    counts = [int(item) for item in num_gt_per_img]
    if len(counts) != batch_size:
        raise ValueError(
            f"num_gt_per_img length {len(counts)} != batch size {batch_size}")
    if any(item < 0 for item in counts):
        raise ValueError("num_gt_per_img cannot contain negative counts")
    if sum(counts) != int(values.shape[0]):
        raise ValueError(
            f"sum(num_gt_per_img) {sum(counts)} != values {values.shape[0]}")
    return counts


def pad_flat_by_batch(values, shape, num_gt_per_img):
    """Turn concatenated per-image values into a padded batch tensor."""
    counts = _validate_batch_layout(values, shape, num_gt_per_img)
    batch_size = int(shape[0])
    max_gt = max(counts, default=0)
    padded = values.new_zeros((batch_size, max_gt, *values.shape[1:]))
    valid = torch.zeros(
        (batch_size, max_gt), dtype=torch.bool, device=values.device)
    if max_gt == 0:
        return padded, valid

    counts_tensor = torch.as_tensor(
        counts, dtype=torch.long, device=values.device)
    batch_ids = torch.repeat_interleave(
        torch.arange(batch_size, device=values.device), counts_tensor,
        output_size=int(values.shape[0]))
    starts = counts_tensor.cumsum(0) - counts_tensor
    local_ids = torch.arange(values.shape[0], device=values.device)
    local_ids = local_ids - torch.repeat_interleave(
        starts, counts_tensor, output_size=int(values.shape[0]))
    padded[batch_ids, local_ids] = values
    valid[batch_ids, local_ids] = True
    return padded, valid


def _normalize_slice_endpoint(endpoint, size):
    """Clip a spatial box endpoint to its raster canvas."""
    return endpoint.clamp(0, size)


def padded_box_coverage(padded_boxes, valid, height, width):
    """Return the pixel coverage of every padded box as [B, M, H, W]."""
    boxes = padded_boxes.long()
    u1 = _normalize_slice_endpoint(boxes[..., 0], width)
    v1 = _normalize_slice_endpoint(boxes[..., 1], height)
    u2 = _normalize_slice_endpoint(boxes[..., 2], width)
    v2 = _normalize_slice_endpoint(boxes[..., 3], height)
    rows = torch.arange(height, device=boxes.device).view(1, 1, height, 1)
    cols = torch.arange(width, device=boxes.device).view(1, 1, 1, width)
    coverage = valid[..., None, None]
    coverage = coverage & (cols >= u1[..., None, None])
    coverage = coverage & (cols < u2[..., None, None])
    coverage = coverage & (rows >= v1[..., None, None])
    coverage = coverage & (rows < v2[..., None, None])
    return coverage


def compute_fg_mask_vectorized(
        gt_boxes2d, shape, num_gt_per_img, downsample_factor=1,
        device=torch.device("cpu")):
    """Vectorized equivalent of ``compute_fg_mask``."""
    requested_device = torch.device(device)
    if requested_device.type != gt_boxes2d.device.type:
        raise ValueError("gt_boxes2d and requested mask must share a device")
    if (requested_device.index is not None
            and requested_device.index != gt_boxes2d.device.index):
        raise ValueError("gt_boxes2d and requested mask must share a device")

    gt_boxes2d /= downsample_factor
    gt_boxes2d[:, :2] = torch.floor(gt_boxes2d[:, :2])
    gt_boxes2d[:, 2:] = torch.ceil(gt_boxes2d[:, 2:])
    boxes = gt_boxes2d.long()
    padded_boxes, valid = pad_flat_by_batch(
        boxes, shape, num_gt_per_img)
    batch_size, height, width = int(shape[0]), int(shape[-2]), int(shape[-1])
    if padded_boxes.shape[1] == 0:
        return torch.zeros(shape, dtype=torch.bool, device=gt_boxes2d.device)
    return padded_box_coverage(
        padded_boxes, valid, height, width).any(dim=1)
