import unittest

import torch

from lib.models.monodgp.monodgp import clip_depth_mean_gradients


class DepthOutputGradientClippingTest(unittest.TestCase):

    def test_forward_is_elementwise_identical(self):
        values = torch.randn(2, 3, 2, requires_grad=True)
        clipped = clip_depth_mean_gradients(values, max_norm=0.03)

        self.assertTrue(torch.equal(clipped, values))

    def test_clips_only_mean_depth_and_preserves_log_uncertainty(self):
        values = torch.zeros(3, 2, requires_grad=True)
        receipts = []
        clipped = clip_depth_mean_gradients(
            values, max_norm=0.03, receipt_sink=receipts)
        upstream = torch.tensor([
            [0.06, 0.40],
            [0.02, -0.50],
            [0.00, 0.90],
        ])

        clipped.backward(upstream)

        expected = torch.tensor([
            [0.03, 0.40],
            [0.02, -0.50],
            [0.00, 0.90],
        ])
        self.assertTrue(torch.allclose(values.grad, expected))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(int(receipts[0]['prediction_count']), 3)
        self.assertEqual(int(receipts[0]['clipped_count']), 1)
        self.assertAlmostEqual(
            float(receipts[0]['pre_clip_max_absolute_gradient']),
            0.06, places=6)
        self.assertAlmostEqual(
            float(receipts[0]['minimum_scale']), 0.5, places=6)
        self.assertAlmostEqual(
            float(receipts[0]['pre_clip_energy']), 0.004, places=6)
        self.assertAlmostEqual(
            float(receipts[0]['post_clip_energy']), 0.0013, places=6)

    def test_infinite_threshold_is_exactly_gradient_equivalent(self):
        values = torch.randn(4, 5, 2, requires_grad=True)
        upstream = torch.randn_like(values)

        clip_depth_mean_gradients(
            values, max_norm=float('inf')).backward(upstream)

        self.assertTrue(torch.equal(values.grad, upstream))

    def test_rejects_invalid_threshold_and_channel_count(self):
        with self.assertRaises(ValueError):
            clip_depth_mean_gradients(torch.zeros(1, 2), max_norm=0)
        with self.assertRaises(ValueError):
            clip_depth_mean_gradients(torch.zeros(1, 3), max_norm=0.03)


if __name__ == '__main__':
    unittest.main()
