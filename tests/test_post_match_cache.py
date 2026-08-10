import unittest

import torch

from lib.models.monodgp.monodgp import build_post_match_cache


LOSSES = ('labels', 'boxes', 'center', 'depths', 'dims', 'angles')
FIELDS = ('labels', 'boxes_3d', 'depth', 'size_3d',
          'heading_bin', 'heading_res')


def strided(rows, columns, device, offset):
    base = torch.arange(
        rows * columns * 2, device=device, dtype=torch.float32).reshape(
            rows, columns * 2)
    return base[:, ::2] + offset


def make_target(rows, device, offset):
    return {
        'labels': strided(rows, 1, device, offset).to(torch.int64),
        'boxes_3d': strided(rows, 6, device, offset),
        'depth': strided(rows, 1, device, offset),
        'size_3d': strided(rows, 3, device, offset) + 1.0,
        'heading_bin': strided(rows, 1, device, offset).to(torch.int64) % 12,
        'heading_res': strided(rows, 1, device, offset),
    }


class PostMatchCacheTest(unittest.TestCase):
    def check_case(self, counts, indices, device):
        targets = [
            make_target(count, device, 1000.0 * batch_index)
            for batch_index, count in enumerate(counts)
        ]
        outputs = {
            'pred_boxes': torch.zeros(
                (len(counts), 16, 6), device=device,
                dtype=torch.float32)}
        cache = build_post_match_cache(outputs, targets, indices, LOSSES)

        expected_batch = torch.cat([
            torch.full_like(source, batch_index)
            for batch_index, (source, _) in enumerate(indices)
        ]).to(device)
        expected_source = torch.cat([
            source for source, _ in indices]).to(device)
        self.assertTrue(torch.equal(
            cache['source_index'][0], expected_batch))
        self.assertTrue(torch.equal(
            cache['source_index'][1], expected_source))
        self.assertEqual(
            cache['matched_count'],
            sum(int(source.numel()) for source, _ in indices))
        for field in FIELDS:
            expected = torch.cat([
                target[field][target_index]
                for target, (_, target_index) in zip(targets, indices)
            ], dim=0)
            actual = cache['matched_targets'][field]
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.device, expected.device)
            self.assertTrue(torch.equal(actual, expected), field)

    def run_contracts(self, device):
        empty = torch.empty(0, dtype=torch.int64)
        self.check_case(
            [0, 3, 1, 0],
            [(empty, empty),
             (torch.tensor([7, 2, 9, 4]), torch.tensor([1, 0, 1, 2])),
             (torch.tensor([5]), torch.tensor([0])),
             (empty, empty)],
            device)
        self.check_case(
            [0] * 8, [(empty, empty) for _ in range(8)], device)
        self.check_case(
            [1, 1, 1],
            [(torch.tensor([batch_index + 1]), torch.tensor([0]))
             for batch_index in range(3)],
            device)

    def test_cpu_contracts(self):
        self.run_contracts(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_contracts(self):
        self.run_contracts(torch.device('cuda'))

    def test_out_of_range_target_rejected(self):
        targets = [make_target(1, torch.device('cpu'), 0.0)]
        outputs = {'pred_boxes': torch.zeros((1, 4, 6))}
        with self.assertRaises(IndexError):
            build_post_match_cache(
                outputs, targets,
                [(torch.tensor([0]), torch.tensor([1]))], LOSSES)

    def test_inconsistent_target_fields_rejected(self):
        target = make_target(2, torch.device('cpu'), 0.0)
        target['depth'] = target['depth'][:1]
        outputs = {'pred_boxes': torch.zeros((1, 4, 6))}
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            build_post_match_cache(
                outputs, [target],
                [(torch.tensor([0]), torch.tensor([0]))], LOSSES)
