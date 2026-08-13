import unittest
from unittest.mock import patch

import torch

from lib.models.monodgp.matcher import HungarianMatcher
from lib.models.monodgp.monodgp import SetCriterion


def make_case(device, sizes, tied=False):
    generator = torch.Generator(device=device).manual_seed(7301)
    batch_size = len(sizes)
    queries = 22
    logits = torch.randn(
        batch_size, queries, 3, generator=generator, device=device)
    centers = torch.rand(
        batch_size, queries, 2, generator=generator, device=device)
    sides = torch.rand(
        batch_size, queries, 4, generator=generator, device=device) * 0.3
    boxes = torch.cat((centers, sides), dim=-1)
    if tied:
        logits.zero_()
        boxes.fill_(0.2)
    targets = []
    for size in sizes:
        labels = torch.randint(
            0, 3, (size,), generator=generator, device=device)
        target_centers = torch.rand(
            size, 2, generator=generator, device=device)
        target_sides = torch.rand(
            size, 4, generator=generator, device=device) * 0.3
        boxes_3d = torch.cat((target_centers, target_sides), dim=-1)
        if tied:
            labels.zero_()
            boxes_3d.fill_(0.2)
        targets.append({
            'labels': labels,
            'boxes': boxes_3d[:, 2:6].clone(),
            'boxes_3d': boxes_3d,
        })
    return {'pred_logits': logits, 'pred_boxes': boxes}, targets


class BatchedSameImageMatcherTest(unittest.TestCase):
    def compare_case(self, device, sizes, tied=False):
        outputs, targets = make_case(device, sizes, tied=tied)
        kwargs = dict(cost_class=2.0, cost_3dcenter=10.0,
                      cost_bbox=5.0, cost_giou=2.0)
        ordinary = HungarianMatcher(**kwargs).to(device)
        batched = HungarianMatcher(
            **kwargs, use_batched_same_image_cost=True).to(device)
        expected = ordinary(outputs, targets, group_num=2)
        prepared = batched.prepare_targets(targets)
        observed = batched(
            outputs, targets, group_num=2, prepared_targets=prepared)
        self.assertEqual(len(expected), len(observed))
        for expected_pair, observed_pair in zip(expected, observed):
            self.assertTrue(torch.equal(expected_pair[0], observed_pair[0]))
            self.assertTrue(torch.equal(expected_pair[1], observed_pair[1]))

    def test_cpu_ragged_exact(self):
        self.compare_case(torch.device('cpu'), [0, 1, 3, 7])

    def test_cpu_ties_exact(self):
        self.compare_case(torch.device('cpu'), [0, 1, 4, 6], tied=True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_ragged_exact(self):
        self.compare_case(torch.device('cuda'), [0, 1, 3, 7])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_ties_exact(self):
        self.compare_case(torch.device('cuda'), [0, 1, 4, 6], tied=True)

    def test_non_divisible_query_count_rejected(self):
        outputs, targets = make_case(torch.device('cpu'), [2, 3])
        matcher = HungarianMatcher(
            use_batched_same_image_cost=True)
        with self.assertRaisesRegex(ValueError, 'not divisible'):
            matcher(outputs, targets, group_num=3)

    def test_inconsistent_target_fields_rejected(self):
        _, targets = make_case(torch.device('cpu'), [2])
        targets[0]['labels'] = targets[0]['labels'][:1]
        matcher = HungarianMatcher(
            use_batched_same_image_cost=True)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            matcher.prepare_targets(targets)

    def test_iou3d_comparison_receipt_tracks_changed_match(self):
        outputs, targets = make_case(torch.device('cpu'), [1], tied=True)
        outputs = {key: value[:, :2] for key, value in outputs.items()}
        outputs.update({
            'pred_depth': torch.zeros(1, 2, 2),
            'pred_3d_dim': torch.zeros(1, 2, 6),
            'pred_angle': torch.zeros(1, 2, 24),
        })
        matcher = HungarianMatcher(
            cost_class=2.0, cost_3dcenter=10.0, cost_bbox=5.0,
            cost_giou=2.0, cost_iou3d=5.0,
            use_batched_same_image_cost=True)
        matcher.collect_iou3d_comparison = True
        fake_iou = torch.tensor([[0.0], [1.0]])
        fake_receipt = {'pair_count': 2, 'exact_pair_count': 1}
        with patch(
                'lib.models.monodgp.matcher.pairwise_iou3d_match_cost',
                return_value=(fake_iou, fake_receipt)):
            matcher.collect_iou3d_comparison = False
            unmonitored_indices = matcher(outputs, targets, group_num=1)
            matcher.collect_iou3d_comparison = True
            indices = matcher(outputs, targets, group_num=1)
        self.assertEqual(
            unmonitored_indices[0][0].tolist(), indices[0][0].tolist())
        self.assertEqual(
            unmonitored_indices[0][1].tolist(), indices[0][1].tolist())
        self.assertEqual(indices[0][0].tolist(), [1])
        receipt = matcher.last_iou3d_receipt
        self.assertEqual(receipt['comparison_count'], 1)
        self.assertEqual(receipt['changed_count'], 1)
        self.assertAlmostEqual(receipt['iou3d_gain_sum'], 1.0)
        self.assertAlmostEqual(receipt['giou2d_delta_sum'], 0.0)
        self.assertAlmostEqual(receipt['class_score_delta_sum'], 0.0)
        self.assertAlmostEqual(receipt['current_iou3d_sum'], 1.0)
        self.assertAlmostEqual(receipt['current_giou2d_sum'], 1.0)
        self.assertAlmostEqual(receipt['current_class_score_sum'], 0.5)

    def test_iou3d_receipts_aggregate_absolute_and_delta_metrics(self):
        receipts = [{
            'comparison_count': 2,
            'changed_count': 1,
            'iou3d_gain_sum': 0.4,
            'giou2d_delta_sum': -0.2,
            'class_score_delta_sum': -0.1,
            'current_iou3d_sum': 1.2,
            'current_giou2d_sum': 1.0,
            'current_class_score_sum': 1.5,
        }, {
            'comparison_count': 3,
            'changed_count': 2,
            'iou3d_gain_sum': 0.6,
            'giou2d_delta_sum': -0.3,
            'class_score_delta_sum': 0.2,
            'current_iou3d_sum': 0.8,
            'current_giou2d_sum': 0.5,
            'current_class_score_sum': 1.0,
        }]
        metrics = SetCriterion._iou3d_matching_metrics(
            receipts, torch.device('cpu'))
        expected = {
            'monitor_iou3d_matching_identity_change_fraction': 0.6,
            'monitor_iou3d_matching_mean_iou3d_gain': 0.2,
            'monitor_iou3d_matching_mean_giou2d_delta': -0.1,
            'monitor_iou3d_matching_mean_gt_class_score_delta': 0.02,
            'monitor_iou3d_matching_current_mean_iou3d': 0.4,
            'monitor_iou3d_matching_current_mean_giou2d': 0.3,
            'monitor_iou3d_matching_current_mean_gt_class_score': 0.5,
        }
        self.assertEqual(set(metrics), set(expected))
        for key, expected_value in expected.items():
            self.assertAlmostEqual(float(metrics[key]), expected_value)
