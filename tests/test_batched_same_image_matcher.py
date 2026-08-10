import unittest

import torch

from lib.models.monodgp.matcher import HungarianMatcher


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
