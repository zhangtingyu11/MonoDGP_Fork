import unittest

import torch

from lib.helpers.optimizer_helper import AdamW


def make_parameters(device):
    generator = torch.Generator(device=device).manual_seed(9173)
    return [
        torch.nn.Parameter(torch.randn(
            shape, device=device, generator=generator))
        for shape in ((17,), (3, 5), (2, 3, 4), (11,))
    ]


def make_optimizer(parameters, foreach, amsgrad=False):
    return AdamW([
        {'params': parameters[:2], 'weight_decay': 0.0},
        {'params': parameters[2:], 'weight_decay': 1e-4},
    ], lr=2.5e-5, betas=(0.9, 0.999), eps=1e-8,
        amsgrad=amsgrad, foreach=foreach)


def install_gradients(parameters, step):
    for index, parameter in enumerate(parameters):
        if (step, index) in {(1, 1), (2, 3), (4, 1)}:
            parameter.grad = None
            continue
        values = torch.arange(
            parameter.numel(), device=parameter.device,
            dtype=parameter.dtype).reshape_as(parameter)
        parameter.grad = (values + 1 + step * 0.25 + index).sin()


class ForeachAdamWTest(unittest.TestCase):
    def assert_equal(self, left_params, right_params, left, right):
        for left_param, right_param in zip(left_params, right_params):
            self.assertTrue(torch.equal(left_param, right_param))
        left_state = left.state_dict()
        right_state = right.state_dict()
        self.assertEqual(left_state['param_groups'], right_state['param_groups'])
        self.assertEqual(set(left_state['state']), set(right_state['state']))
        for key in left_state['state']:
            for field, left_value in left_state['state'][key].items():
                right_value = right_state['state'][key][field]
                if torch.is_tensor(left_value):
                    self.assertTrue(torch.equal(left_value, right_value))
                else:
                    self.assertEqual(left_value, right_value)

    def run_equivalence(self, device, amsgrad):
        ordinary_params = make_parameters(device)
        foreach_params = [
            torch.nn.Parameter(value.detach().clone())
            for value in ordinary_params]
        ordinary = make_optimizer(ordinary_params, False, amsgrad)
        foreach = make_optimizer(foreach_params, True, amsgrad)
        for step in range(6):
            install_gradients(ordinary_params, step)
            install_gradients(foreach_params, step)
            ordinary.step()
            foreach.step()
            self.assert_equal(
                ordinary_params, foreach_params, ordinary, foreach)

    def test_cpu_exact(self):
        self.run_equivalence(torch.device('cpu'), amsgrad=False)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_exact(self):
        self.run_equivalence(torch.device('cuda'), amsgrad=False)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_amsgrad_exact(self):
        self.run_equivalence(torch.device('cuda'), amsgrad=True)

    def test_sparse_gradient_rejected(self):
        parameter = torch.nn.Parameter(torch.ones(4))
        optimizer = AdamW([parameter], foreach=True)
        parameter.grad = torch.sparse_coo_tensor(
            torch.tensor([[0, 2]]), torch.tensor([1.0, 1.0]), (4,))
        with self.assertRaisesRegex(RuntimeError, 'sparse gradients'):
            optimizer.step()
