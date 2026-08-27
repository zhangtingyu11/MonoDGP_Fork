import torch

from lib.models.monodgp.iou3d_match_cost import (
    batched_pairwise_iou3d_match_cost,
    pairwise_iou3d_match_cost,
)


def _fixture():
    generator = torch.Generator().manual_seed(444)
    batch_size, query_count = 3, 17
    boxes = torch.rand(batch_size, query_count, 6, generator=generator)
    boxes[..., :2] = boxes[..., :2] * 0.6 + 0.2
    depth = torch.rand(batch_size, query_count, 2, generator=generator)
    depth[..., 0] = depth[..., 0] * 35.0 + 5.0
    dimensions = torch.rand(
        batch_size, query_count, 3, generator=generator) + 1.0
    angle = torch.randn(
        batch_size, query_count, 24, generator=generator)
    outputs = {
        'pred_logits': torch.randn(
            batch_size, query_count, 3, generator=generator),
        'pred_boxes': boxes,
        'pred_depth': depth,
        'pred_3d_dim': dimensions,
        'pred_angle': angle,
    }
    calib = torch.tensor([
        [700.0, 0.0, 640.0, 0.0],
        [0.0, 700.0, 192.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    targets = []
    for count in (2, 5, 3):
        target_boxes = torch.rand(count, 6, generator=generator)
        target_boxes[:, :2] = target_boxes[:, :2] * 0.6 + 0.2
        targets.append({
            'labels': torch.ones(count, dtype=torch.long),
            'boxes_3d': target_boxes,
            'depth': torch.rand(count, generator=generator) * 35.0 + 5.0,
            'depth_unit_scale': torch.ones(count),
            'src_size_3d': torch.rand(
                count, 3, generator=generator) + 1.0,
            'projective_rotation_y': torch.rand(
                count, generator=generator) * 6.0 - 3.0,
            'projective_input_size': torch.tensor([1280.0, 384.0]),
            'projective_image_effective_calib': calib,
            'physical_ray_heading': torch.tensor(True),
        })
    return outputs, targets


def test_batched_cost_is_bitwise_equal_to_per_image_cost():
    outputs, targets = _fixture()
    means = torch.zeros(3, 3)
    actual, actual_receipt = batched_pairwise_iou3d_match_cost(
        outputs, targets, means)
    expected = torch.zeros_like(actual)
    pair_count = 0
    exact_pair_count = 0
    for batch_index, target in enumerate(targets):
        image_outputs = {
            key: value[batch_index] for key, value in outputs.items()
        }
        image_result, receipt = pairwise_iou3d_match_cost(
            image_outputs, target, means)
        expected[batch_index, :, :len(target['labels'])] = image_result
        pair_count += receipt['pair_count']
        exact_pair_count += receipt['exact_pair_count']

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual_receipt == {
        'pair_count': pair_count,
        'exact_pair_count': exact_pair_count,
    }
