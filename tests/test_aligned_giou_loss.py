import unittest

import torch

from utils.box_ops import generalized_box_iou, generalized_box_iou_aligned


def random_valid_boxes(count, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    upper_left = torch.rand((count, 2), generator=generator, device=device)
    size = torch.rand((count, 2), generator=generator, device=device) + 0.01
    return torch.cat((upper_left, upper_left + size), dim=1)


class AlignedGiouLossTest(unittest.TestCase):
    def run_equivalence(self, device):
        for count in (0, 1, 2, 17, 193):
            left = random_valid_boxes(count, device, 1000 + count)
            right = random_valid_boxes(count, device, 2000 + count)
            left.requires_grad_(True)
            right.requires_grad_(True)
            pairwise = torch.diag(generalized_box_iou(left, right))
            aligned = generalized_box_iou_aligned(left, right)
            self.assertTrue(torch.equal(pairwise, aligned))
            if count:
                pairwise_grad = torch.autograd.grad(
                    pairwise.sum(), (left, right), retain_graph=True)
                aligned_grad = torch.autograd.grad(
                    aligned.sum(), (left, right))
                for old, new in zip(pairwise_grad, aligned_grad):
                    self.assertTrue(torch.equal(old, new))

    def test_cpu_output_and_gradient_exact(self):
        self.run_equivalence(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_output_and_gradient_exact(self):
        self.run_equivalence(torch.device('cuda'))

    def test_shape_contract(self):
        with self.assertRaisesRegex(ValueError, 'equal box shapes'):
            generalized_box_iou_aligned(
                torch.zeros(2, 4), torch.zeros(3, 4))
        with self.assertRaisesRegex(ValueError, r'\[N, 4\]'):
            generalized_box_iou_aligned(
                torch.zeros(2, 5), torch.zeros(2, 5))

    def test_degenerate_contract(self):
        valid = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        invalid = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
        with self.assertRaises(AssertionError):
            generalized_box_iou_aligned(invalid, valid)
