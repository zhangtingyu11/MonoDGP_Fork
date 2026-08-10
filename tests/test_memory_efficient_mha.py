import copy
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class _ZeroDeformableAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, query, *args, **kwargs):
        return torch.zeros_like(query)


def _load_transformer_module(name, relative_path):
    root = Path(__file__).resolve().parents[1]
    package = sys.modules.setdefault("isolated_monodgp", types.ModuleType(
        "isolated_monodgp"))
    package.__path__ = []
    ops = sys.modules.setdefault(
        "isolated_monodgp.ops", types.ModuleType("isolated_monodgp.ops"))
    ops.__path__ = []
    modules = types.ModuleType("isolated_monodgp.ops.modules")
    modules.MSDeformAttn = _ZeroDeformableAttention
    modules.MSDeformAttn_cross = _ZeroDeformableAttention
    modules.MultiheadAttention = nn.MultiheadAttention
    sys.modules["isolated_monodgp.ops.modules"] = modules
    spec = importlib.util.spec_from_file_location(
        f"isolated_monodgp.{name}", root / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


depth_transformer = _load_transformer_module(
    "depth_transformer",
    "lib/models/monodgp/depth_predictor/transformer.py")
det2d_transformer = _load_transformer_module(
    "det2d_transformer", "lib/models/monodgp/det2d_transformer.py")
det3d_transformer = _load_transformer_module(
    "det3d_transformer", "lib/models/monodgp/det3d_transformer.py")
TransformerEncoderLayer = depth_transformer.TransformerEncoderLayer
TransformerDecoderLayer = det2d_transformer.TransformerDecoderLayer
DepthAwareDecoderLayer = det3d_transformer.DepthAwareDecoderLayer


def _paired(module):
    control = copy.deepcopy(module)
    candidate = copy.deepcopy(module)
    control.use_memory_efficient_mha = False
    candidate.use_memory_efficient_mha = True
    control.eval()
    candidate.eval()
    return control, candidate


class MemoryEfficientMhaContract(unittest.TestCase):
    def assert_close(self, control, candidate):
        torch.testing.assert_close(
            control, candidate, rtol=2e-6, atol=2e-7)

    def test_depth_encoder_forward_and_input_gradient(self):
        torch.manual_seed(7)
        base = TransformerEncoderLayer(
            16, nhead=4, dim_feedforward=32, dropout=0.0)
        control, candidate = _paired(base)
        src_control = torch.randn(9, 2, 16, requires_grad=True)
        src_candidate = src_control.detach().clone().requires_grad_(True)
        pos = torch.randn(9, 2, 16)
        mask = torch.zeros(2, 9, dtype=torch.bool)
        mask[:, -2:] = True
        out_control = control(src_control, mask, pos)
        out_candidate = candidate(src_candidate, mask, pos)
        self.assert_close(out_control, out_candidate)
        out_control.square().mean().backward()
        out_candidate.square().mean().backward()
        self.assert_close(src_control.grad, src_candidate.grad)

    def test_2d_decoder_self_attention_forward_and_gradient(self):
        torch.manual_seed(11)
        base = TransformerDecoderLayer(
            d_model=16, d_ffn=32, dropout=0.0, n_levels=1,
            n_heads=4, n_points=1, group_num=1)
        base.cross_attn = _ZeroDeformableAttention()
        control, candidate = _paired(base)
        tgt_control = torch.randn(2, 5, 16, requires_grad=True)
        tgt_candidate = tgt_control.detach().clone().requires_grad_(True)
        query_pos = torch.randn(2, 5, 16)
        args = (
            query_pos,
            torch.rand(2, 5, 1, 2),
            torch.randn(2, 3, 16),
            torch.tensor([[1, 3]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.zeros(2, 3, dtype=torch.bool),
            2,
        )
        out_control = control(tgt_control, *args)
        out_candidate = candidate(tgt_candidate, *args)
        self.assert_close(out_control, out_candidate)
        out_control.square().mean().backward()
        out_candidate.square().mean().backward()
        self.assert_close(tgt_control.grad, tgt_candidate.grad)

    def test_3d_decoder_depth_and_self_attention_forward_and_gradient(self):
        torch.manual_seed(13)
        base = DepthAwareDecoderLayer(
            d_model=16, d_ffn=32, dropout=0.0, n_levels=1,
            n_heads=4, n_points=1, group_num=1)
        base.cross_attn = _ZeroDeformableAttention()
        control, candidate = _paired(base)
        tgt_control = torch.randn(2, 5, 16, requires_grad=True)
        tgt_candidate = tgt_control.detach().clone().requires_grad_(True)
        args = (
            torch.randn(2, 5, 16),
            torch.rand(2, 5, 1, 2),
            torch.randn(2, 3, 16),
            torch.tensor([[1, 3]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.zeros(2, 3, dtype=torch.bool),
            torch.randn(7, 2, 16),
            torch.zeros(2, 7, dtype=torch.bool),
            2,
        )
        out_control = control(tgt_control, *args)
        out_candidate = candidate(tgt_candidate, *args)
        self.assert_close(out_control, out_candidate)
        out_control.square().mean().backward()
        out_candidate.square().mean().backward()
        self.assert_close(tgt_control.grad, tgt_candidate.grad)


if __name__ == "__main__":
    unittest.main()
