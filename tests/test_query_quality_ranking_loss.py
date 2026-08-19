from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from lib.helpers.config_helper import load_config
from lib.helpers.quality_ranking_monitor import QualityRankingAccumulator
from lib.losses.query_quality_ranking_loss import (
    all_query_quality_ranking_loss)
from lib.models.monodgp.monodgp import SetCriterion


ROOT = Path(__file__).resolve().parents[1]


def test_all_queries_receive_pointwise_max_iou_supervision():
    prediction = torch.tensor(
        [[[0.2], [-0.4], [0.8], [-0.9]]], requires_grad=True)
    iou = torch.tensor([[[0.6, 0.2],
                         [0.1, 0.4],
                         [0.9, 0.3],
                         [0.0, 0.0]]])

    result = all_query_quality_ranking_loss(
        prediction, iou, (2,), group_num=1,
        ranking_iou_gap=0.1, low_iou_threshold=0.1,
        low_iou_weight=0.1)

    expected_target = torch.tensor([0.6, 0.4, 0.9, 0.0])
    decoded = ((prediction[..., 0] + 1.0) * 0.5).clamp(0, 1)
    torch.testing.assert_close(
        result['monitor_quality_target_iou_mean'],
        expected_target.mean())
    torch.testing.assert_close(
        result['monitor_quality_iou_mae'],
        (decoded.detach()[0] - expected_target).abs().mean())
    assert result['loss_quality_point'].isfinite()


def test_ranking_pairs_stay_inside_group_and_assigned_gt():
    prediction = torch.tensor(
        [[[-0.5], [0.5], [0.0], [0.8], [0.7], [-0.2]]],
        requires_grad=True)
    # Two groups of three. Only q0/q1 form a valid same-group, same-GT pair:
    # q3/q4 have an IoU gap below 0.1; all other pairs change GT or group.
    iou = torch.tensor([[[0.8, 0.1],
                         [0.6, 0.2],
                         [0.1, 0.75],
                         [0.9, 0.0],
                         [0.85, 0.0],
                         [0.0, 0.4]]])

    result = all_query_quality_ranking_loss(
        prediction, iou, (2,), group_num=2,
        ranking_iou_gap=0.1)

    assert result['monitor_quality_rank_pair_count'].item() == 1
    assert result['monitor_quality_rank_pair_accuracy'].item() == 0
    result['loss_quality_rank'].backward()
    # Gradient descent must increase q0 and decrease q1 because 0.8 > 0.6.
    assert prediction.grad[0, 0, 0] < 0
    assert prediction.grad[0, 1, 0] > 0
    torch.testing.assert_close(
        prediction.grad[0, 2:, 0], torch.zeros(4))


def test_iou_target_is_detached_but_quality_prediction_is_not():
    prediction = torch.zeros((1, 4, 1), requires_grad=True)
    iou = torch.tensor([[[0.8], [0.6], [0.4], [0.0]]], requires_grad=True)

    result = all_query_quality_ranking_loss(
        prediction, iou, (1,), group_num=1)
    total = result['loss_quality_point'] + 0.2 * result['loss_quality_rank']
    total.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert iou.grad is None


def test_empty_target_image_has_background_point_loss_and_no_rank_pairs():
    prediction = torch.zeros((1, 4, 1), requires_grad=True)
    iou = torch.empty((1, 4, 0))

    result = all_query_quality_ranking_loss(
        prediction, iou, (0,), group_num=2)

    # SmoothL1(0, -1)=0.5; the configured low-IoU weight must remain an
    # absolute 0.1 multiplier instead of being cancelled by normalization.
    torch.testing.assert_close(
        result['loss_quality_point'], torch.tensor(0.05))
    assert result['loss_quality_rank'].item() == 0
    assert result['monitor_quality_rank_pair_count'].item() == 0
    result['loss_quality_point'].backward()
    assert torch.isfinite(prediction.grad).all()


def test_point_weight_ramps_continuously_from_iou_0_1_to_0_5():
    prediction = torch.tensor(
        [[[-0.8], [-0.6], [-0.2], [0.0]]], requires_grad=True)
    iou = torch.tensor([[[0.1], [0.2], [0.4], [0.5]]])

    result = all_query_quality_ranking_loss(
        prediction, iou, (1,), group_num=1,
        low_iou_threshold=0.1, low_iou_weight=0.1,
        full_weight_iou=0.5)

    expected_weights = torch.tensor([0.1, 0.325, 0.775, 1.0])
    torch.testing.assert_close(
        result['monitor_quality_point_effective_weight_mean'],
        expected_weights.mean())
    # Predictions equal their encoded IoU targets, so the ramp itself must
    # not introduce any artificial loss at either boundary.
    torch.testing.assert_close(
        result['loss_quality_point'], torch.tensor(0.0))


def test_quality_ranking_configuration_validation():
    prediction = torch.zeros((1, 5, 1))
    iou = torch.zeros((1, 5, 1))

    try:
        all_query_quality_ranking_loss(
            prediction, iou, (1,), group_num=2)
    except ValueError as error:
        assert 'divisible' in str(error)
    else:
        raise AssertionError('invalid query grouping was accepted')


def test_experiment32_changes_only_quality_supervision_from_experiment31():
    exp31 = load_config(ROOT / 'configs/monodgp_exp31.yaml')
    exp32 = load_config(ROOT / 'configs/monodgp_exp32.yaml')

    assert exp32['dataset'] == exp31['dataset']
    for section in ('optimizer', 'lr_scheduler', 'tester'):
        assert exp32[section] == exp31[section]
    exp31_model = dict(exp31['model'])
    exp32_model = dict(exp32['model'])
    exp31_quality = exp31_model.pop('iou_quality_head')
    exp32_quality = exp32_model.pop('iou_quality_head')
    assert exp32_model == exp31_model
    assert exp32_quality['enabled'] is True
    assert exp32_quality['init_seed'] == exp31_quality['init_seed']
    assert exp32_quality['target_encoding'] == exp31_quality['target_encoding']
    assert exp32_quality['supervision'] == 'all_query_same_gt_ranking'
    assert exp32_quality['point_loss_coef'] == 1.0
    assert exp32_quality['rank_loss_coef'] == 0.2
    assert exp32_quality['ranking_iou_gap'] == 0.1
    assert exp32_quality['low_iou_threshold'] == 0.1
    assert exp32_quality['low_iou_weight'] == 0.1
    assert exp32_quality['full_weight_iou'] == 0.5
    exp31_trainer = dict(exp31['trainer'])
    exp32_trainer = dict(exp32['trainer'])
    exp31_trainer.pop('swanlab')
    exp32_trainer.pop('swanlab')
    assert exp32_trainer == exp31_trainer


class _QualityOnlyMatcher(nn.Module):
    def __init__(self, iou3d):
        super().__init__()
        self.iou3d = iou3d
        self.last_iou3d_matrix = None
        self.last_iou3d_receipt = {}
        self.cost_iou3d = 5.0
        self.collect_iou3d_comparison = False
        self.use_batched_same_image_cost = False

    def forward(self, outputs, targets, group_num=11,
                prepared_targets=None):
        del group_num, prepared_targets
        self.last_iou3d_matrix = self.iou3d.to(
            outputs['pred_quality'].device)
        empty = torch.empty(
            0, dtype=torch.long, device=outputs['pred_quality'].device)
        return [(empty, empty) for _ in targets]


def test_criterion_emits_and_weights_final_and_auxiliary_quality_losses():
    iou = torch.tensor([[[0.8], [0.6], [0.3], [0.0]]])
    matcher = _QualityOnlyMatcher(iou)
    weight_dict = {
        'loss_quality_point': 1.0,
        'loss_quality_rank': 0.2,
        'loss_quality_point_0': 1.0,
        'loss_quality_rank_0': 0.2,
        'loss_quality_point_1': 1.0,
        'loss_quality_rank_1': 0.2,
    }
    with patch('torch.cuda.current_device', return_value=0):
        criterion = SetCriterion(
            num_classes=3, matcher=matcher, weight_dict=weight_dict,
            focal_alpha=0.25, losses=['quality'], inter_losses=[],
            group_num=2, query_monitoring={'enabled': False},
            iou_quality_head={
                'enabled': True,
                'supervision': 'all_query_same_gt_ranking',
                'ranking_iou_gap': 0.1,
                'low_iou_threshold': 0.1,
                'low_iou_weight': 0.1,
                'full_weight_iou': 0.5,
            })
    predictions = [
        torch.zeros((1, 4, 1), requires_grad=True) for _ in range(3)]
    outputs = {
        'pred_quality': predictions[-1],
        'pred_logits': torch.zeros((1, 4, 3)),
        'inter_outputs': [],
        'aux_outputs': [
            {'pred_quality': predictions[0]},
            {'pred_quality': predictions[1]},
        ],
    }
    targets = [{'labels': torch.tensor([1])}]

    losses = criterion(outputs, targets)

    expected = {
        'loss_quality_point', 'loss_quality_rank',
        'loss_quality_point_0', 'loss_quality_rank_0',
        'loss_quality_point_1', 'loss_quality_rank_1',
    }
    assert expected <= losses.keys()
    total = sum(losses[key] * weight_dict[key] for key in expected)
    total.backward()
    assert all(prediction.grad is not None for prediction in predictions)
    assert all(torch.isfinite(prediction.grad).all()
               for prediction in predictions)


def test_validation_monitor_reports_direct_ranking_decision_metrics():
    accumulator = QualityRankingAccumulator(
        [{'name': 'fused', 'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0}])
    accumulator.query_iou = [torch.tensor([0.8, 0.6, 0.2]).numpy()]
    scores = {
        'classification': torch.tensor([0.7, 0.9, 0.1]).numpy(),
        'quality': torch.tensor([0.9, 0.7, 0.1]).numpy(),
        'depth': torch.tensor([0.8, 0.6, 0.2]).numpy(),
        'fused': torch.tensor([0.9, 0.8, 0.1]).numpy(),
    }
    accumulator.query_scores = {
        name: [value] for name, value in scores.items()}
    accumulator.oracle_rows = [{
        'difficulty': 'moderate',
        'distance': 25.0,
        'occlusion': 1,
        'one_to_one_query': 0,
        'one_to_one_iou': 0.8,
        'best_query': 0,
        'best_iou': 0.8,
        'target_iou': torch.tensor([0.8, 0.6, 0.2]).numpy(),
        'scores': scores,
    }]

    quality = accumulator.finalize()['one_to_one_oracle']['all']['quality']

    assert quality['top1_identity_fraction'] == 1.0
    assert quality['top3_identity_fraction'] == 1.0
    assert abs(quality['top1_iou_regret']) < 1e-7
    assert quality['pairwise_order_pair_count_gap_ge_0_1'] == 3
    assert quality['pairwise_order_accuracy_gap_ge_0_1'] == 1.0
    assert quality['high_quality_top1_iou_ge_0_7_fraction'] == 1.0
