"""Exact deterministic indexed backward for RegionSegHead 2x upsampling."""

from itertools import product

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable


_FORCE_DETERMINISTIC_BILINEAR_BACKWARD = False


def set_force_deterministic_bilinear_backward(enabled):
    global _FORCE_DETERMINISTIC_BILINEAR_BACKWARD
    _FORCE_DETERMINISTIC_BILINEAR_BACKWARD = bool(enabled)


class _DeterministicIndexedCorner(Function):
    @staticmethod
    def forward(ctx, value, y_index, x_index, corner):
        ctx.input_height = value.shape[-2]
        ctx.input_width = value.shape[-1]
        ctx.corner = corner
        return torch.ops.aten._unsafe_index.Tensor(
            value, [None, None, y_index, x_index])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_value):
        from .ms_deform_attn_func import ensure_deterministic_msda_available

        extension = ensure_deterministic_msda_available()
        grad_input = extension.deterministic_bilinear2d_index_backward(
            grad_value.contiguous(),
            ctx.input_height,
            ctx.input_width,
            ctx.corner,
        )
        return grad_input, None, None, None


def _source_indices(input_size, output_size, nsqueeze, value):
    scale = (input_size - 1.0) / (output_size - 1.0)
    source = torch.arange(output_size, device=value.device).to(
        dtype=value.dtype)
    source = (source * scale).clamp(min=0.0)
    source = source.reshape(source.shape[0], *([1] * nsqueeze))
    base = source.to(torch.int64)
    following = (base + 1).clamp(max=input_size - 1)
    weight = (source - base).clamp(0.0, 1.0).to(value.dtype)
    return base, following, weight


def _deterministic_bilinear_upsample2x(value):
    input_height, input_width = value.shape[-2:]
    output_height = input_height * 2
    output_width = input_width * 2
    y, yp1, y_weight = _source_indices(
        input_height, output_height, 1, value)
    x, xp1, x_weight = _source_indices(
        input_width, output_width, 0, value)
    indices = ((y, yp1), (x, xp1))
    values = []
    for corner, choices in enumerate(product((0, 1), repeat=2)):
        values.append(_DeterministicIndexedCorner.apply(
            value,
            indices[0][choices[0]],
            indices[1][choices[1]],
            corner,
        ))
    top = values[0] + torch.mul(values[1] - values[0], x_weight)
    bottom = values[2] + torch.mul(values[3] - values[2], x_weight)
    result = top + torch.mul(bottom - top, y_weight)
    return result.contiguous()


def deterministic_bilinear_upsample2x(value):
    if _FORCE_DETERMINISTIC_BILINEAR_BACKWARD:
        return _deterministic_bilinear_upsample2x(value)
    return F.interpolate(
        value, scale_factor=2.0, mode="bilinear", align_corners=True)
