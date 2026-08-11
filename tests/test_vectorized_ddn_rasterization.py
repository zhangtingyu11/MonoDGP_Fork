import unittest

import torch

from lib.models.monodgp.depth_predictor.ddn_loss.balancer import (
    compute_fg_mask,
    compute_fg_mask_vectorized,
)
from lib.models.monodgp.depth_predictor.ddn_loss.ddn_loss import DDNLoss


class VectorizedDDNRasterizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required by the existing DDNLoss")
        cls.device = torch.device("cuda")

    def compare_case(self, boxes, depths, counts, height, width):
        boxes = boxes.to(self.device)
        depths = depths.to(self.device)
        logits = torch.zeros(
            (len(counts), 81, height, width),
            dtype=torch.float32,
            device=self.device)
        legacy_boxes = boxes.clone()
        vector_boxes = boxes.clone()
        legacy = DDNLoss(use_vectorized_rasterization=False)
        vector = DDNLoss(use_vectorized_rasterization=True)

        legacy_map = legacy.build_target_depth_from_3dcenter(
            logits, legacy_boxes, depths.clone(), counts)
        vector_map = vector.build_target_depth_from_3dcenter(
            logits, vector_boxes, depths.clone(), counts)
        torch.testing.assert_close(
            legacy_map, vector_map, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(
            legacy_boxes, vector_boxes, rtol=0, atol=0, equal_nan=True)

        legacy_fg = compute_fg_mask(
            legacy_boxes, legacy_map.shape, counts, device=self.device)
        vector_fg = compute_fg_mask_vectorized(
            vector_boxes, vector_map.shape, counts, device=self.device)
        self.assertTrue(torch.equal(legacy_fg, vector_fg))
        torch.testing.assert_close(
            legacy_boxes, vector_boxes, rtol=0, atol=0, equal_nan=True)

    def test_overlap_empty_and_clipped_canvas_boundaries(self):
        boxes = torch.tensor([
            [-1.2, -1.2, 3.1, 3.1],
            [-7.0, -6.0, 10.0, 9.0],
            [-8.0, 0.0, -1.0, 4.0],
            [6.0, 5.0, 8.0, 6.0],
            [4.8, 2.2, 2.1, 4.9],
            [0.0, 0.0, 0.0, 5.0],
            [1.2, 1.2, 6.8, 5.8],
            [2.0, 2.0, 5.0, 5.0],
            [2.0, 2.0, 5.0, 5.0],
        ], dtype=torch.float32)
        depths = torch.tensor(
            [30.0, 20.0, 10.0, 0.0, 5.0, 7.0, 18.0, 12.0, 3.0])
        self.compare_case(boxes, depths, [6, 3], height=6, width=8)
        self.compare_case(
            torch.empty((0, 4)), torch.empty((0,)), [0, 0], 4, 5)

    def test_negative_start_keeps_visible_canvas_region(self):
        boxes = torch.tensor(
            [[-2.0, 1.0, 3.0, 5.0]], dtype=torch.float32,
            device=self.device)
        depths = torch.tensor([7.0], device=self.device)
        logits = torch.zeros(
            (1, 81, 6, 8), dtype=torch.float32, device=self.device)

        for vectorized in (False, True):
            working_boxes = boxes.clone()
            loss = DDNLoss(use_vectorized_rasterization=vectorized)
            depth_map = loss.build_target_depth_from_3dcenter(
                logits, working_boxes, depths.clone(), [1])
            expected_depth = torch.zeros_like(depth_map)
            expected_depth[:, 1:5, 0:3] = 7.0
            torch.testing.assert_close(
                depth_map, expected_depth, rtol=0, atol=0)

            foreground = (
                compute_fg_mask_vectorized
                if vectorized else compute_fg_mask)(
                    working_boxes, depth_map.shape, [1], device=self.device)
            expected_foreground = torch.zeros_like(foreground)
            expected_foreground[:, 1:5, 0:3] = True
            self.assertTrue(torch.equal(foreground, expected_foreground))

    def test_fully_negative_box_does_not_wrap_from_canvas_end(self):
        boxes = torch.tensor(
            [[-7.0, 1.0, -1.0, 5.0]], dtype=torch.float32,
            device=self.device)
        depths = torch.tensor([7.0], device=self.device)
        logits = torch.zeros(
            (1, 81, 6, 8), dtype=torch.float32, device=self.device)

        for vectorized in (False, True):
            working_boxes = boxes.clone()
            loss = DDNLoss(use_vectorized_rasterization=vectorized)
            depth_map = loss.build_target_depth_from_3dcenter(
                logits, working_boxes, depths.clone(), [1])
            self.assertFalse((depth_map > 0).any())

            foreground = (
                compute_fg_mask_vectorized
                if vectorized else compute_fg_mask)(
                    working_boxes, depth_map.shape, [1], device=self.device)
            self.assertFalse(foreground.any())

    def test_randomized_exact_equivalence(self):
        generator = torch.Generator().manual_seed(20260810)
        for _ in range(100):
            batch_size = int(torch.randint(1, 5, (), generator=generator))
            counts = torch.randint(
                0, 8, (batch_size,), generator=generator).tolist()
            total = sum(counts)
            height = int(torch.randint(2, 14, (), generator=generator))
            width = int(torch.randint(2, 18, (), generator=generator))
            boxes = torch.randn(
                (total, 4), generator=generator) * max(height, width)
            if total:
                boxes[::5, 2:] = boxes[::5, :2]
            depths = torch.rand((total,), generator=generator) * 60
            self.compare_case(boxes, depths, counts, height, width)

    def test_full_loss_and_gradient_exact_equivalence(self):
        generator = torch.Generator(device=self.device).manual_seed(8181)
        base_logits = torch.randn(
            (2, 81, 6, 8), generator=generator, device=self.device)
        boxes = torch.tensor([
            [-1.2, -1.2, 5.8, 4.1],
            [1.1, 1.3, 7.4, 5.9],
            [2.0, 2.0, 6.0, 6.0],
        ], dtype=torch.float32, device=self.device)
        depths = torch.tensor([30.0, 15.0, 4.0], device=self.device)
        legacy_logits = base_logits.clone().requires_grad_(True)
        vector_logits = base_logits.clone().requires_grad_(True)
        legacy_loss = DDNLoss(use_vectorized_rasterization=False)(
            legacy_logits, boxes.clone(), [2, 1], depths.clone())
        vector_loss = DDNLoss(use_vectorized_rasterization=True)(
            vector_logits, boxes.clone(), [2, 1], depths.clone())
        legacy_loss.backward()
        vector_loss.backward()
        torch.testing.assert_close(legacy_loss, vector_loss, rtol=0, atol=0)
        torch.testing.assert_close(
            legacy_logits.grad, vector_logits.grad, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
