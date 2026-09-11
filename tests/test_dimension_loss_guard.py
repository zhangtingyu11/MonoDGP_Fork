import json
import logging

import pytest
import torch
import torch.nn.functional as F

from lib.helpers.dimension_loss_audit import DimensionLossAudit
from lib.models.monodgp.monodgp import SetCriterion


def candidate(prediction, target):
    count = target.shape[0]
    cache = {
        'source_index': (torch.zeros(count, dtype=torch.long, device=target.device),
                         torch.arange(count, device=target.device)),
        'matched_targets': {'size_3d': target},
    }
    return SetCriterion.loss_dims(None, {'pred_3d_dim': prediction}, [], [],
                                  max(count, 1), matched_cache=cache)


def historical(prediction, target):
    src = prediction[0]
    loss = torch.abs(src - target)
    loss /= target.clone().detach()
    with torch.no_grad():
        compensation = F.l1_loss(src, target) / loss.mean()
    loss *= compensation
    return loss.sum() / target.shape[0]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('delta', [0.1, 1e-5])
def test_normal_loss_and_gradient_are_bitwise_unchanged(device, dtype, delta):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA visibility required')
    target = torch.tensor([[1.5, 1.6, 4.0], [1.7, 1.8, 4.2]], device=device, dtype=dtype)
    offsets = torch.tensor([[delta, -delta, 0], [-delta, 0, delta]], device=device, dtype=dtype)
    old = (target + offsets).unsqueeze(0).requires_grad_()
    new = old.detach().clone().requires_grad_()
    before = historical(old, target)
    result = candidate(new, target)
    before.backward(); result['loss_dim'].backward()
    assert torch.equal(before, result['loss_dim'])
    assert torch.equal(old.grad, new.grad)
    assert result['monitor_dim_zero_guard'].item() == 0
    assert result['monitor_dim_empty_guard'].item() == 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('empty', [False, True])
def test_zero_and_empty_loss_have_finite_zero_gradients(device, empty):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA visibility required')
    target = torch.empty((0, 3), device=device) if empty else torch.tensor([[1.5, 1.6, 4.0]], device=device)
    prediction = target.unsqueeze(0).clone().requires_grad_()
    result = candidate(prediction, target)
    result['loss_dim'].backward()
    assert result['loss_dim'].item() == 0
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))
    assert result['monitor_dim_zero_guard'].item() == int(not empty)
    assert result['monitor_dim_empty_guard'].item() == int(empty)


def test_audit_records_every_batch_and_separates_diagnostic_triggers(tmp_path):
    baseline = tmp_path / 'baseline.log'
    baseline.write_text('Train metrics: epoch=1/250, step=1/232, image_ids=[4], '
                        'lr=[0.0002], loss_detr=1.000000, losses={loss_dim=1.000000}\n')
    audit = DimensionLossAudit(tmp_path, baseline, logging.getLogger(__name__))
    losses = {'loss_dim': torch.tensor(1.), 'monitor_dim_zero_guard': torch.tensor(0.),
              'monitor_dim_empty_guard': torch.tensor(0.),
              'monitor_group0_monitor_dim_zero_guard': torch.tensor(1.)}
    audit.observe(1, 1, losses, 1., [4], [.0002])
    audit.observe(1, 2, losses, 1., [5], [.0002])
    losses['loss_dim'] = torch.tensor(2.)
    audit.observe(1, 1, losses, 2., [4], [.0002])
    audit.save_summary(1, 3)
    summary = json.loads((audit.directory / 'summary.json').read_text())
    assert summary['observed_training_batches'] == 3
    assert summary['compared_batches'] == 2
    assert summary['mismatched_batches'] == 1
    assert summary['guard_triggers_by_key']['monitor_dim_zero_guard'] == 0
    assert summary['guard_triggers_by_key']['monitor_group0_monitor_dim_zero_guard'] == 3
    assert len((audit.directory / 'batches.jsonl').read_text().splitlines()) == 3
