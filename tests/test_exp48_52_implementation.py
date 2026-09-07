from copy import deepcopy
import math
from pathlib import Path
import os
import subprocess
import sys
import types

import numpy as np
import pytest
import torch
from PIL import Image

# Dataset-only tests must not initialize the historical GPU evaluator as an
# import side effect in CPU-only pytest collection.  On a CUDA-visible host,
# keep the real module available for the evaluator tests collected alongside
# this file.
_EVAL_MODULE = 'lib.datasets.kitti.kitti_eval_python.eval'
if not torch.cuda.is_available() and _EVAL_MODULE not in sys.modules:
    _eval_stub = types.ModuleType(_EVAL_MODULE)
    _eval_stub.get_official_eval_result = lambda *_args, **_kwargs: None
    _eval_stub.get_distance_eval_result = lambda *_args, **_kwargs: None
    sys.modules[_EVAL_MODULE] = _eval_stub

from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.datasets.kitti.kitti_utils import affine_transform
from lib.datasets.kitti.kitti_utils import get_affine_transform
from lib.helpers.config_helper import load_config
from lib.helpers.decode_helper import fused_quality_score
from lib.helpers.optimizer_helper import AdamW as MonoDGPAdamW
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.tester_helper import Tester as MonoDGPTester
from lib.models.monodgp.monodgp import SetCriterion


ROOT = Path(__file__).resolve().parents[1]


def _config(number):
    return load_config(ROOT / f'configs/monodgp_exp{number}.yaml')


def _without_metadata(config):
    config = deepcopy(config)
    config.pop('model_name')
    config['trainer'].pop('swanlab')
    return config


def _synthetic_kitti_root(tmp_path, label=None, include_donor=False):
    root = tmp_path / 'kitti'
    for relative in (
            'ImageSets', 'training/image_2', 'training/calib',
            'training/label_2'):
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / 'ImageSets/train.txt').write_text(
        '000000\n000001\n' if include_donor else '000000\n',
        encoding='utf-8')
    (root / 'ImageSets/val.txt').write_text(
        '000000\n', encoding='utf-8')

    x = np.arange(1280, dtype=np.uint16)[None, :]
    y = np.arange(384, dtype=np.uint16)[:, None]
    image = np.stack((
        np.broadcast_to(x % 256, (384, 1280)),
        np.broadcast_to(y % 256, (384, 1280)),
        (x + y) % 256,
    ), axis=-1).astype(np.uint8)
    Image.fromarray(image).save(root / 'training/image_2/000000.png')
    if include_donor:
        Image.fromarray(255 - image).save(
            root / 'training/image_2/000001.png')

    calibration = '\n'.join((
        'P0: 700 0 640 0 0 700 192 0 0 0 1 0',
        'P1: 700 0 640 0 0 700 192 0 0 0 1 0',
        'P2: 700 0 640 0 0 700 192 0 0 0 1 0',
        'P3: 700 0 640 0 0 700 192 0 0 0 1 0',
        'R0_rect: 1 0 0 0 1 0 0 0 1',
        'Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0',
    )) + '\n'
    (root / 'training/calib/000000.txt').write_text(
        calibration, encoding='utf-8')
    if label is None:
        label = (
            'Car 0 0 0 550 160 730 276.5 '
            '1.5 1.6 4.0 0 1.5 20 0\n')
    (root / 'training/label_2/000000.txt').write_text(
        label, encoding='utf-8')
    if include_donor:
        (root / 'training/calib/000001.txt').write_text(
            calibration, encoding='utf-8')
        (root / 'training/label_2/000001.txt').write_text(
            label, encoding='utf-8')
    return root, image


def _dataset_config(number, root, random_mixup):
    cfg = deepcopy(_config(number)['dataset'])
    cfg.update({
        'root_dir': str(root),
        'aug_pd': False,
        'aug_crop': False,
        'random_flip': 0.0,
        'random_crop': 0.0,
        'random_mixup3d': float(random_mixup),
    })
    return cfg


def _assert_projected_center_matches_encoded_target(targets, slot=0):
    center = np.array([0.0, 0.75, 20.0, 1.0], dtype=np.float32)
    homogeneous = targets['calibs'][slot] @ center
    projected = homogeneous[:2] / homogeneous[2]
    encoded = targets['boxes_3d'][slot, :2] * np.array(
        [1280.0, 384.0], dtype=np.float32)
    np.testing.assert_allclose(projected, encoded, rtol=0, atol=2e-5)


def _expected_virtual_focal_input(source_image, dataset, multiplier):
    _, inverse = get_affine_transform(
        np.array([640.0, 192.0]),
        np.array([1280.0, 384.0]) / multiplier,
        0, np.array([1280, 384]), inv=1)
    expected = Image.fromarray(source_image).transform(
        (1280, 384), method=Image.AFFINE,
        data=tuple(inverse.reshape(-1).tolist()),
        resample=Image.BILINEAR)
    expected = np.asarray(expected).astype(np.float32) / 255.0
    return ((expected - dataset.mean) / dataset.std).transpose(2, 0, 1)


def _assert_affine_2d_box_matches_targets(targets, multiplier, slot=0):
    transform = get_affine_transform(
        np.array([640.0, 192.0]),
        np.array([1280.0, 384.0]) / multiplier,
        0, np.array([1280, 384]))
    expected_box = np.array([550.0, 160.0, 730.0, 276.5])
    expected_box[:2] = affine_transform(expected_box[:2], transform)
    expected_box[2:] = affine_transform(expected_box[2:], transform)
    resolution = np.array([1280.0, 384.0])
    expected_center_size = np.concatenate((
        (expected_box[:2] + expected_box[2:]) * 0.5 / resolution,
        (expected_box[2:] - expected_box[:2]) / resolution,
    ))
    np.testing.assert_allclose(
        targets['boxes'][slot], expected_center_size, rtol=0, atol=2e-7)

    center = targets['boxes_3d'][slot, :2]
    left, right, top, bottom = targets['boxes_3d'][slot, 2:]
    reconstructed = np.array([
        center[0] - left, center[1] - top,
        center[0] + right, center[1] + bottom,
    ])
    np.testing.assert_allclose(
        reconstructed,
        expected_box / np.array([1280.0, 384.0, 1280.0, 384.0]),
        rtol=0, atol=2e-7)


def test_exp48_is_only_the_approved_single_iou_classification_change():
    control = _without_metadata(_config(47))
    candidate = _without_metadata(_config(48))

    quality = candidate['model'].pop('iou_quality_head')
    control_quality = control['model'].pop('iou_quality_head')
    iou_classification = candidate['model'].pop('iou_classification')
    negative_weighting = candidate['model'][
        'high_iou_unmatched_negative_weighting']
    tester = candidate.pop('tester')
    control.pop('tester')

    assert quality['enabled'] is False
    assert {key: value for key, value in quality.items() if key != 'enabled'} == {
        key: value for key, value in control_quality.items()
        if key != 'enabled'
    }
    assert iou_classification == {'enabled': True, 'beta': 2.0}
    assert negative_weighting['enabled'] is False
    assert tester['primary_quality_score'] == 'iou_classification_x_depth'
    assert tester['quality_score_fusions'] == [{
        'name': 'iou_classification_x_depth',
        'alpha': 1.0,
        'beta': 0.0,
        'gamma': 1.0,
        'historical_topk': True,
    }]
    assert tester['quality_ranking_monitoring'] is False
    assert candidate == control


@pytest.mark.parametrize(
    ('number', 'scope'), ((49, 'all_samples'), (50, 'successful_mixup')))
def test_exp49_and_exp50_only_add_the_approved_virtual_focal_scope(
        number, scope):
    control = _without_metadata(_config(47))
    candidate = _without_metadata(_config(number))
    dataset = candidate['dataset']
    assert dataset.pop('mixup_virtual_focal') is True
    assert dataset.pop('mixup_virtual_focal_scope') == scope
    assert dataset.pop('mixup_virtual_focal_multipliers') == [0.9, 1.0, 1.1]
    assert candidate == control


def test_exp49_applies_virtual_focal_without_mixup_and_syncs_rgb_p2_targets(
        tmp_path):
    root, source_image = _synthetic_kitti_root(tmp_path)
    dataset = KITTI_Dataset(
        'train', _dataset_config(49, root, random_mixup=0.0))
    dataset.set_epoch(0)

    inputs, model_calib, targets, _ = dataset[0]

    assert targets['mixup_requested'] == 0.0
    assert targets['mixup_applied'] == 0.0
    assert targets['mixup_virtual_focal_eligible'] == 1.0
    assert targets['mixup_virtual_focal_requested_multiplier'] == pytest.approx(
        0.9)
    assert targets['mixup_virtual_focal_multiplier'] == pytest.approx(0.9)
    assert targets['mixup_virtual_focal_cancelled'] == 0.0
    assert model_calib[0, 0] == pytest.approx(630.0, abs=1e-4)
    assert model_calib[1, 1] == pytest.approx(630.0, abs=1e-4)
    _assert_projected_center_matches_encoded_target(targets)
    _assert_affine_2d_box_matches_targets(targets, 0.9)

    expected = _expected_virtual_focal_input(source_image, dataset, 0.9)
    np.testing.assert_allclose(inputs, expected, rtol=0, atol=0)


def test_exp50_applies_virtual_focal_only_after_successful_same_p2_mixup(
        tmp_path, monkeypatch):
    root, source_image = _synthetic_kitti_root(
        tmp_path, include_donor=True)

    no_mix = KITTI_Dataset(
        'train', _dataset_config(50, root, random_mixup=0.0))
    no_mix.set_epoch(0)
    _, no_mix_calib, no_mix_targets, _ = no_mix[0]
    assert no_mix_targets['mixup_applied'] == 0.0
    assert no_mix_targets['mixup_virtual_focal_eligible'] == 0.0
    assert no_mix_targets['mixup_virtual_focal_multiplier'] == 1.0
    assert no_mix_calib[0, 0] == pytest.approx(700.0, abs=1e-4)

    successful = KITTI_Dataset(
        'train', _dataset_config(50, root, random_mixup=1.0))
    successful.set_epoch(0)
    monkeypatch.setattr(np.random, 'choice', lambda _values: '000001')
    mixed_inputs, mixed_calib, mixed_targets, _ = successful[0]
    assert mixed_targets['mixup_requested'] == 1.0
    assert mixed_targets['mixup_applied'] == 1.0
    assert mixed_targets['mixup_cross_focal'] == 0.0
    assert mixed_targets['mixup_donor_index'] == 1
    assert mixed_targets['mixup_virtual_focal_eligible'] == 1.0
    assert mixed_targets['mixup_virtual_focal_multiplier'] == pytest.approx(
        0.9)
    assert mixed_calib[0, 0] == pytest.approx(630.0, abs=1e-4)
    _assert_projected_center_matches_encoded_target(mixed_targets)
    _assert_projected_center_matches_encoded_target(mixed_targets, slot=1)
    _assert_affine_2d_box_matches_targets(mixed_targets, 0.9)
    _assert_affine_2d_box_matches_targets(mixed_targets, 0.9, slot=1)
    assert mixed_targets['mask_2d'][:2].tolist() == [True, True]
    assert mixed_targets['mixup_is_donor'][:2].tolist() == [False, True]
    blended = np.asarray(Image.blend(
        Image.fromarray(source_image),
        Image.fromarray(255 - source_image), alpha=0.5))
    expected_mixed = _expected_virtual_focal_input(
        blended, successful, 0.9)
    np.testing.assert_allclose(mixed_inputs, expected_mixed, rtol=0, atol=0)
    assert not np.array_equal(
        mixed_inputs,
        _expected_virtual_focal_input(source_image, successful, 0.9))


def test_exp50_failed_donor_search_restores_multiplier_one(tmp_path):
    root, _ = _synthetic_kitti_root(tmp_path)
    dataset = KITTI_Dataset(
        'train', _dataset_config(50, root, random_mixup=1.0))
    dataset.max_objs = 1
    dataset.set_epoch(0)
    np.random.seed(11)

    _, model_calib, targets, _ = dataset[0]

    assert targets['mixup_requested'] == 1.0
    assert targets['mixup_applied'] == 0.0
    assert targets['mixup_attempts'] == 50.0
    assert targets['mixup_virtual_focal_eligible'] == 0.0
    assert targets['mixup_virtual_focal_multiplier'] == 1.0
    assert model_calib[0, 0] == pytest.approx(700.0, abs=1e-4)


def test_exp49_canvas_cut_fallback_and_validation_disable_augmentation(
        tmp_path):
    boundary_label = (
        'Car 0 0 0 40 160 120 276.5 '
        '1.5 1.6 4.0 -16 1.5 20 0\n')
    root, _ = _synthetic_kitti_root(tmp_path, label=boundary_label)
    cfg = _dataset_config(49, root, random_mixup=0.0)

    train = KITTI_Dataset('train', cfg)
    train.set_epoch(2)
    _, train_calib, train_targets, _ = train[0]
    assert train_targets['mixup_virtual_focal_requested_multiplier'] == (
        pytest.approx(1.1))
    assert train_targets['mixup_virtual_focal_cancelled'] == 1.0
    assert train_targets['mixup_virtual_focal_multiplier'] == 1.0
    assert train_calib[0, 0] == pytest.approx(700.0, abs=1e-4)

    validation = KITTI_Dataset('val', cfg)
    validation.set_epoch(2)
    _, val_calib, val_targets, _ = validation[0]
    assert val_targets['mixup_virtual_focal_eligible'] == 0.0
    assert val_targets['mixup_virtual_focal_multiplier'] == 1.0
    assert val_calib[0, 0] == pytest.approx(700.0, abs=1e-4)


def test_exp50_successful_mixup_uses_same_canvas_cut_fallback(tmp_path):
    boundary_label = (
        'Car 0 0 0 40 160 120 276.5 '
        '1.5 1.6 4.0 -16 1.5 20 0\n')
    root, source_image = _synthetic_kitti_root(
        tmp_path, label=boundary_label)
    dataset = KITTI_Dataset(
        'train', _dataset_config(50, root, random_mixup=1.0))
    dataset.set_epoch(2)
    np.random.seed(13)

    inputs, model_calib, targets, _ = dataset[0]

    assert targets['mixup_applied'] == 1.0
    assert targets['mixup_virtual_focal_eligible'] == 1.0
    assert targets['mixup_virtual_focal_requested_multiplier'] == (
        pytest.approx(1.1))
    assert targets['mixup_virtual_focal_cancelled'] == 1.0
    assert targets['mixup_virtual_focal_multiplier'] == 1.0
    assert model_calib[0, 0] == pytest.approx(700.0, abs=1e-4)
    np.testing.assert_allclose(
        inputs,
        _expected_virtual_focal_input(source_image, dataset, 1.0),
        rtol=0, atol=0)


def test_exp51_delays_quality_loss_and_formal_q_scoring_to_epoch_121():
    control = _without_metadata(_config(47))
    candidate = _without_metadata(_config(51))

    quality = candidate['model'].pop('iou_quality_head')
    control_quality = control['model'].pop('iou_quality_head')
    assert quality.pop('training_start_epoch') == 121
    assert quality.pop('enabled') is True
    assert control_quality.pop('enabled') is False
    assert quality == control_quality
    assert candidate['tester']['primary_quality_score'] == 'c_q_d'
    assert candidate['tester']['quality_score_fusions'] == [{
        'name': 'c_q_d',
        'alpha': 1.0,
        'beta': 1.0,
        'gamma': 1.0,
    }]
    for key in (
            'quality_score_activation_epoch',
            'pre_activation_primary_quality_score',
            'pre_activation_quality_score_fusions'):
        candidate['tester'].pop(key)
    candidate['tester']['primary_quality_score'] = control['tester'][
        'primary_quality_score']
    candidate['tester']['quality_score_fusions'] = control['tester'][
        'quality_score_fusions']
    candidate['trainer']['validation_start_epoch'] = 120
    assert candidate == control


def test_exp52_only_replaces_step_lr_with_the_approved_cosine_schedule():
    control = _without_metadata(_config(47))
    candidate = _without_metadata(_config(52))
    assert candidate.pop('lr_scheduler') == {
        'type': 'cos',
        'warmup': True,
        'decay_rate': 0.5,
        'decay_list': [85, 125, 165, 205],
        'warmup_type': 'linear',
        'warmup_epochs': 10,
        'warmup_init_lr': 1e-6,
        't_max': 240,
        'eta_min': 1e-6,
    }
    control.pop('lr_scheduler')
    assert candidate == control


def test_iou_classification_times_depth_does_not_require_quality_head():
    outputs = {
        'pred_logits': torch.tensor([[[0.0], [math.log(3.0)]]]),
        'pred_depth': torch.tensor([[[10.0, 0.0], [20.0, math.log(2.0)]]]),
    }
    score = fused_quality_score(
        outputs, alpha=1.0, beta=0.0, gamma=1.0)
    torch.testing.assert_close(
        score, torch.tensor([[[0.5], [0.375]]]), rtol=0, atol=3e-8)


def test_existing_quality_head_fusion_keeps_historical_operation_order():
    outputs = {
        'pred_logits': torch.randn(2, 4, 3),
        'pred_depth': torch.randn(2, 4, 2),
        'pred_quality': torch.randn(2, 4, 1),
    }
    classification = outputs['pred_logits'].sigmoid()
    quality = ((outputs['pred_quality'] + 1.0) * 0.5).clamp(0, 1)
    depth = torch.exp(-outputs['pred_depth'][:, :, 1:2]).clamp_min(0)
    expected = (classification.clamp_min(1e-12).pow(0.25)
                * quality.clamp_min(1e-12).pow(2.0)
                * depth.clamp_min(1e-12).pow(0.75))
    actual = fused_quality_score(
        outputs, alpha=0.25, beta=2.0, gamma=0.75)
    assert torch.equal(actual, expected)


def test_quality_head_has_no_gradient_before_activation():
    criterion = object.__new__(SetCriterion)
    torch.nn.Module.__init__(criterion)
    criterion.iou_quality_training_start_epoch = 121
    criterion.current_training_epoch = 120
    criterion.iou_quality_supervision = 'hungarian_positive'
    logits = torch.nn.Parameter(torch.tensor([[[1.0]]]))
    quality = torch.nn.Parameter(torch.tensor([[[0.25]]]))
    optimizer = MonoDGPAdamW(
        [logits, quality], lr=0.1, weight_decay=0.1)
    quality_before = quality.detach().clone()

    result = criterion.loss_quality(
        {'pred_logits': logits, 'pred_quality': quality}, [], [], 1.0)
    (logits.sum() + result['loss_quality']).backward()

    assert result['loss_quality'].item() == 0.0
    assert quality.grad is None
    torch.testing.assert_close(logits.grad, torch.ones_like(logits))
    optimizer.step()
    assert torch.equal(quality, quality_before)


def test_tester_switches_from_cxd_to_cqd_at_epoch_121():
    tester = object.__new__(MonoDGPTester)
    tester.quality_score_activation_epoch = 121
    tester.pre_activation_quality_score_specs = ({'name': 'current_cxd'},)
    tester.pre_activation_primary_quality_score = 'current_cxd'
    tester.quality_score_specs = ({'name': 'c_q_d'},)
    tester.primary_quality_score = 'c_q_d'

    tester.current_epoch = 120
    assert tester._active_quality_scoring() == (
        ({'name': 'current_cxd'},), 'current_cxd')
    tester.current_epoch = 121
    assert tester._active_quality_scoring() == (
        ({'name': 'c_q_d'},), 'c_q_d')


def test_evaluate_only_restores_exp51_scoring_epoch_from_checkpoint(
        tmp_path, monkeypatch):
    checkpoint = tmp_path / 'checkpoint_best.pth'
    checkpoint.touch()
    observed = {}

    def fake_load_checkpoint(**_kwargs):
        return 120, 0.0, 0

    tester = object.__new__(MonoDGPTester)
    tester.cfg = {'mode': 'single'}
    tester.train_cfg = {
        'save_all': False,
        'primary_ap_only_validation': True,
    }
    tester.output_dir = str(tmp_path)
    tester.device = torch.device('cpu')
    tester.logger = types.SimpleNamespace(info=lambda *_args: None)
    tester.model = torch.nn.Identity()
    tester.quality_score_activation_epoch = 121
    tester.pre_activation_quality_score_specs = ({'name': 'current_cxd'},)
    tester.pre_activation_primary_quality_score = 'current_cxd'
    tester.quality_score_specs = ({'name': 'c_q_d'},)
    tester.primary_quality_score = 'c_q_d'
    tester.current_epoch = None
    tester.inference = lambda **_kwargs: observed.setdefault(
        'score', tester._active_quality_scoring()[1]) or {}
    tester.evaluate = lambda _results: None
    monkeypatch.setattr(
        'lib.helpers.tester_helper.load_checkpoint', fake_load_checkpoint)

    tester.test()

    assert tester.current_epoch == 120
    assert observed['score'] == 'current_cxd'


def test_cosine_scheduler_matches_closed_form():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    scheduler, warmup = build_lr_scheduler({
        'type': 'cos',
        'warmup': False,
        't_max': 250,
        'eta_min': 0.0,
    }, optimizer, last_epoch=-1)
    assert warmup is None
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    for epoch in range(1, 251):
        optimizer.step()
        scheduler.step()
        expected = 2e-4 * 0.5 * (
            1.0 + math.cos(math.pi * epoch / 250))
        assert optimizer.param_groups[0]['lr'] == pytest.approx(
            expected, rel=0, abs=1e-15)


def test_exp52_combines_ten_epoch_linear_warmup_with_240_epoch_cosine():
    cfg = _config(52)['lr_scheduler']
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    scheduler, warmup = build_lr_scheduler(
        cfg, optimizer, last_epoch=-1)

    assert isinstance(warmup, torch.optim.lr_scheduler._LRScheduler)
    assert warmup.num_epoch == 10
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-6)
    observed = {0: optimizer.param_groups[0]['lr']}
    for epoch_index in range(250):
        optimizer.step()
        if epoch_index < warmup.num_epoch:
            warmup.step()
        else:
            scheduler.step()
        human_epoch = epoch_index + 1
        if human_epoch in (1, 10, 11, 250):
            observed[human_epoch] = optimizer.param_groups[0]['lr']

    assert observed[0] == pytest.approx(1e-6, rel=0, abs=1e-15)
    assert observed[1] == pytest.approx(2.09e-5, rel=0, abs=1e-15)
    assert observed[10] == pytest.approx(2e-4, rel=0, abs=1e-15)
    expected_e11 = 1e-6 + (2e-4 - 1e-6) * 0.5 * (
        1.0 + math.cos(math.pi / 240))
    assert observed[11] == pytest.approx(expected_e11, rel=0, abs=1e-15)
    assert observed[250] == pytest.approx(1e-6, rel=0, abs=1e-15)


@pytest.mark.parametrize('checkpoint_epoch', (5, 125))
def test_exp52_warmup_and_cosine_resume_matches_uninterrupted_exactly(
        tmp_path, checkpoint_epoch):
    cfg = _config(52)['lr_scheduler']

    def build():
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=2e-4)
        scheduler, warmup = build_lr_scheduler(
            cfg, optimizer, last_epoch=-1)
        return parameter, optimizer, scheduler, warmup

    parameter, optimizer, scheduler, warmup = build()
    checkpoint = None
    for epoch_index in range(250):
        optimizer.step()
        if epoch_index < warmup.num_epoch:
            warmup.step()
        else:
            scheduler.step()
        if epoch_index + 1 == checkpoint_epoch:
            checkpoint = get_checkpoint_state(
                model=None, optimizer=optimizer,
                lr_scheduler=scheduler,
                warmup_lr_scheduler=warmup,
                epoch=checkpoint_epoch)
    uninterrupted_lr = optimizer.param_groups[0]['lr']
    uninterrupted_scheduler = deepcopy(scheduler.state_dict())
    uninterrupted_warmup = deepcopy(warmup.state_dict())

    path = tmp_path / 'checkpoint.pth'
    torch.save(checkpoint, path)
    resumed_parameter, resumed_optimizer, resumed_scheduler, resumed_warmup = (
        build())
    load_checkpoint(
        model=None, optimizer=resumed_optimizer, filename=str(path),
        map_location='cpu',
        logger=types.SimpleNamespace(info=lambda *_args: None),
        lr_scheduler=resumed_scheduler,
        warmup_lr_scheduler=resumed_warmup)
    for epoch_index in range(checkpoint_epoch, 250):
        resumed_optimizer.step()
        if epoch_index < resumed_warmup.num_epoch:
            resumed_warmup.step()
        else:
            resumed_scheduler.step()

    assert resumed_optimizer.param_groups[0]['lr'] == uninterrupted_lr
    assert resumed_scheduler.state_dict() == uninterrupted_scheduler
    assert resumed_warmup.state_dict() == uninterrupted_warmup
    del parameter, resumed_parameter


@pytest.mark.parametrize(('scheduler_type', 'checkpoint_epoch'), (
    ('cos', 125),
    ('step', 84),
    ('step', 124),
))
def test_scheduler_checkpoint_resume_matches_uninterrupted_training_exactly(
        tmp_path, scheduler_type, checkpoint_epoch):
    scheduler_cfg = {
        'type': scheduler_type,
        'warmup': False,
        'decay_rate': 0.5,
        'decay_list': [85, 125, 165, 205],
        't_max': 250,
        'eta_min': 0.0,
    }

    def build():
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=2e-4)
        scheduler, warmup = build_lr_scheduler(
            scheduler_cfg, optimizer, last_epoch=-1)
        return parameter, optimizer, scheduler, warmup

    parameter, optimizer, scheduler, warmup = build()
    for _ in range(checkpoint_epoch):
        optimizer.step()
        scheduler.step()
    path = tmp_path / 'checkpoint'
    torch.save(get_checkpoint_state(
        model=None, optimizer=optimizer, epoch=checkpoint_epoch,
        best_result=0.0, best_epoch=0,
        lr_scheduler=scheduler, warmup_lr_scheduler=warmup),
        path.with_suffix('.pth'))

    resumed_parameter, resumed_optimizer, resumed_scheduler, resumed_warmup = (
        build())
    load_checkpoint(
        model=None, optimizer=resumed_optimizer,
        filename=path.with_suffix('.pth'), map_location='cpu',
        logger=types.SimpleNamespace(info=lambda *_args: None),
        lr_scheduler=resumed_scheduler,
        warmup_lr_scheduler=resumed_warmup)
    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    resumed_optimizer.step()
    resumed_scheduler.step()

    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    scheduler.step()
    assert resumed_optimizer.param_groups[0]['lr'] == (
        optimizer.param_groups[0]['lr'])
    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    del resumed_parameter


@pytest.mark.parametrize(('scheduler_type', 'checkpoint_epoch'), (
    ('cos', 125),
    ('step', 84),
    ('step', 124),
))
def test_legacy_checkpoint_resume_reconstructs_exact_scheduler_epoch(
        tmp_path, scheduler_type, checkpoint_epoch):
    scheduler_cfg = {
        'type': scheduler_type,
        'warmup': False,
        'decay_rate': 0.5,
        'decay_list': [85, 125, 165, 205],
        't_max': 250,
        'eta_min': 0.0,
    }

    def build():
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=2e-4)
        scheduler, _ = build_lr_scheduler(
            scheduler_cfg, optimizer, last_epoch=-1)
        return parameter, optimizer, scheduler

    parameter, optimizer, scheduler = build()
    for _ in range(checkpoint_epoch):
        optimizer.step()
        scheduler.step()
    checkpoint = get_checkpoint_state(
        model=None, optimizer=optimizer, epoch=checkpoint_epoch,
        best_result=0.0, best_epoch=0)
    checkpoint.pop('lr_scheduler_state')
    checkpoint.pop('warmup_lr_scheduler_state')
    path = tmp_path / 'legacy.pth'
    torch.save(checkpoint, path)

    resumed_parameter, resumed_optimizer, resumed_scheduler = build()
    load_checkpoint(
        model=None, optimizer=resumed_optimizer, filename=path,
        map_location='cpu',
        logger=types.SimpleNamespace(info=lambda *_args: None),
        lr_scheduler=resumed_scheduler)
    assert resumed_scheduler.last_epoch == checkpoint_epoch
    assert resumed_optimizer.param_groups[0]['lr'] == (
        optimizer.param_groups[0]['lr'])
    resumed_optimizer.step()
    resumed_scheduler.step()

    optimizer.step()
    scheduler.step()
    assert resumed_optimizer.param_groups[0]['lr'] == (
        optimizer.param_groups[0]['lr'])
    del parameter, resumed_parameter


def test_step_scheduler_keeps_exp46_milestones_unchanged():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    scheduler, _ = build_lr_scheduler({
        'type': 'step',
        'warmup': False,
        'decay_rate': 0.5,
        'decay_list': [85, 125, 165, 205],
    }, optimizer, last_epoch=-1)

    observed = {}
    for epoch in range(1, 207):
        optimizer.step()
        scheduler.step()
        if epoch in (84, 85, 124, 125, 164, 165, 204, 205):
            observed[epoch] = optimizer.param_groups[0]['lr']
    assert observed == {
        84: 2e-4,
        85: 1e-4,
        124: 1e-4,
        125: 5e-5,
        164: 5e-5,
        165: 2.5e-5,
        204: 2.5e-5,
        205: 1.25e-5,
    }


def test_formal_runner_reports_tee_failure_in_atomic_status(tmp_path):
    fake_root = tmp_path / 'MonoDGP'
    fake_python = fake_root / '.venv-cu129/bin/python'
    fake_python.parent.mkdir(parents=True)
    (fake_root / 'outputs').mkdir()
    output = fake_root / 'outputs/V2-0048_实验48_全Query三维IoU分类乘深度'
    fake_python.write_text(
        '#!/usr/bin/env bash\n'
        'if [[ "$1" == tools/write_run_manifest.py ]]; then\n'
        f'  mkdir -p "{output}"\n'
        'fi\n'
        'exit 0\n',
        encoding='utf-8')
    fake_python.chmod(0o755)
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    fake_tee = fake_bin / 'tee'
    fake_tee.write_text('#!/usr/bin/env bash\nexit 7\n', encoding='utf-8')
    fake_tee.chmod(0o755)

    source = (ROOT / 'tools/run_prepared_exp48_52.sh').read_text(
        encoding='utf-8')
    source = source.replace(
        'ROOT=/home/zhangtingyu/Project/Mono3D/MonoDGP',
        f'ROOT={fake_root}')
    runner = tmp_path / 'runner.sh'
    runner.write_text(source, encoding='utf-8')
    environment = dict(os.environ)
    environment['TMUX'] = 'test-tmux'
    environment['PATH'] = f'{fake_bin}:{environment["PATH"]}'

    completed = subprocess.run(
        ('bash', str(runner), '48'), env=environment,
        text=True, capture_output=True, check=False)

    assert completed.returncode == 95
    assert (output / 'status.tsv').read_text(encoding='utf-8') == (
        'runner_exit\t95\nmanifest_exit\t0\n'
        'train_exit\t0\ntee_exit\t7\n')
    assert not (output / 'status.tsv.tmp').exists()


def test_formal_runner_records_manifest_failure_atomically(tmp_path):
    fake_root = tmp_path / 'MonoDGP'
    fake_python = fake_root / '.venv-cu129/bin/python'
    fake_python.parent.mkdir(parents=True)
    (fake_root / 'outputs').mkdir()
    output = fake_root / 'outputs/V2-0048_实验48_全Query三维IoU分类乘深度'
    fake_python.write_text('#!/usr/bin/env bash\nexit 7\n', encoding='utf-8')
    fake_python.chmod(0o755)
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    source = (ROOT / 'tools/run_prepared_exp48_52.sh').read_text(
        encoding='utf-8').replace(
            'ROOT=/home/zhangtingyu/Project/Mono3D/MonoDGP',
            f'ROOT={fake_root}')
    runner = tmp_path / 'runner.sh'
    runner.write_text(source, encoding='utf-8')
    environment = dict(os.environ)
    environment['TMUX'] = 'test-tmux'
    environment['PATH'] = f'{fake_bin}:{environment["PATH"]}'

    completed = subprocess.run(
        ('bash', str(runner), '48'), env=environment,
        text=True, capture_output=True, check=False)

    assert completed.returncode == 7
    assert (output / 'status.tsv').read_text(encoding='utf-8') == (
        'runner_exit\t7\nmanifest_exit\t7\n'
        'train_exit\t-1\ntee_exit\t-1\n')
    assert not list(output.glob('status.tsv.tmp.*'))


def test_relay_stops_when_exp47_tmux_disappears_and_refuses_duplicate(
        tmp_path):
    fake_root = tmp_path / 'MonoDGP'
    (fake_root / 'outputs').mkdir(parents=True)
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    fake_tmux = fake_bin / 'tmux'
    fake_tmux.write_text('#!/usr/bin/env bash\nexit 1\n', encoding='utf-8')
    fake_tmux.chmod(0o755)

    source = (ROOT / 'tools/relay_exp47_to_exp52.sh').read_text(
        encoding='utf-8')
    source = source.replace(
        'ROOT=/home/zhangtingyu/Project/Mono3D/MonoDGP',
        f'ROOT={fake_root}')
    relay = tmp_path / 'relay.sh'
    relay.write_text(source, encoding='utf-8')
    environment = dict(os.environ)
    environment['TMUX'] = 'test-tmux'
    environment['PATH'] = f'{fake_bin}:{environment["PATH"]}'

    first = subprocess.run(
        ('bash', str(relay)), env=environment,
        text=True, capture_output=True, check=False)
    status = fake_root / 'outputs/exp47_to_exp52_relay/status.tsv'
    assert first.returncode == 92
    assert 'blocked_exp47_session_missing_without_status' in (
        status.read_text(encoding='utf-8'))

    original_status = status.read_text(encoding='utf-8')
    second = subprocess.run(
        ('bash', str(relay)), env=environment,
        text=True, capture_output=True, check=False)
    assert second.returncode == 94
    assert status.read_text(encoding='utf-8') == original_status
