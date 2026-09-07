"""GPU preflight for prepared Exp48-52 without training runs or SwanLab."""

from copy import deepcopy
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numba
import torch
import torchvision


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.config_helper import load_config
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.model_helper import build_model
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import collect_mixup_counts
from lib.helpers.utils_helper import set_random_seed


EXPECTED = {
    'executable': str(ROOT / '.venv-cu129/bin/python'),
    'python': '3.10.20',
    'torch': '2.8.0+cu129',
    'torchvision': '0.23.0+cu129',
    'cuda': '12.9',
    'cudnn': 91002,
    'numba': '0.66.0',
    'numba_cuda': '0.30.4',
}


def _runtime_receipt():
    actual = {
        'executable': sys.executable,
        'python': '.'.join(map(str, sys.version_info[:3])),
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'numba': numba.__version__,
        'numba_cuda': importlib.metadata.version('numba-cuda'),
    }
    mismatches = {
        key: {'expected': EXPECTED[key], 'actual': actual[key]}
        for key in EXPECTED if actual[key] != EXPECTED[key]
    }
    if mismatches:
        raise RuntimeError(f'preflight runtime mismatch: {mismatches}')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('preflight requires exactly one visible CUDA GPU')
    return actual


def _preflight_config(number):
    cfg = load_config(ROOT / f'configs/monodgp_exp{number}.yaml')
    cfg = deepcopy(cfg)
    cfg['dataset']['batch_size'] = 1
    return cfg


def _enable_exp46_execution_contract():
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory = False
    from lib.models.monodgp.ops.functions.ms_deform_attn_func import (
        ensure_deterministic_msda_available,
        set_force_deterministic_msda,
    )
    from lib.models.monodgp.ops.functions.deterministic_bilinear import (
        set_force_deterministic_bilinear_backward,
    )
    set_force_deterministic_msda(True)
    ensure_deterministic_msda_available()
    set_force_deterministic_bilinear_backward(True)


def _move_batch_to_cuda(batch):
    inputs, calibs, targets, info = batch
    inputs = inputs.cuda(non_blocking=False)
    calibs = calibs.cuda(non_blocking=False)
    targets = {
        key: value.cuda(non_blocking=False)
        for key, value in targets.items()
    }
    return inputs, calibs, targets, info


def _prepare_targets(targets, batch_size):
    mask = targets['mask_2d']
    object_keys = {
        'labels', 'boxes', 'calibs', 'depth', 'size_3d',
        'heading_bin', 'heading_res', 'boxes_3d', 'src_size_3d',
        'depth_unit_scale', 'projective_rotation_y', 'mixup_is_donor',
    }
    passthrough_keys = {
        'img_size', 'projective_input_size',
        'projective_image_effective_calib', 'physical_ray_heading',
    }
    prepared = []
    for batch_index in range(batch_size):
        item = {}
        for key, value in targets.items():
            if key in object_keys:
                item[key] = value[batch_index][mask[batch_index]]
            elif key in ('depth_map', 'obj_region'):
                item[key] = value[batch_index]
            elif key in passthrough_keys:
                item[key] = value[batch_index]
        prepared.append(item)
    return prepared


def _weighted_loss(losses, weight_dict, prefix=None):
    terms = []
    for key, value in losses.items():
        if key not in weight_dict:
            continue
        if prefix is not None and not key.startswith(prefix):
            continue
        terms.append(value * weight_dict[key])
    if not terms:
        raise RuntimeError(f'no weighted loss terms for prefix={prefix!r}')
    total = sum(terms)
    if not torch.isfinite(total):
        raise RuntimeError(f'non-finite weighted loss for prefix={prefix!r}')
    return total


def _release(*values):
    del values
    gc.collect()
    torch.cuda.empty_cache()


def _data_receipt(number):
    cfg = _preflight_config(number)
    set_random_seed(cfg.get('random_seed', 444))
    train_loader, val_loader = build_dataloader(cfg['dataset'], workers=4)
    train_loader.dataset.set_epoch(0)
    train_batch = next(iter(train_loader))
    counts = collect_mixup_counts(train_batch[2])
    if not counts:
        raise RuntimeError(f'Exp{number} emitted no MixUp receipt')
    targets = train_batch[2]
    eligible = targets['mixup_virtual_focal_eligible'].bool()
    applied = targets['mixup_applied'].bool()
    multiplier = targets['mixup_virtual_focal_multiplier'].float()
    requested = targets['mixup_virtual_focal_requested_multiplier'].float()
    allowed = torch.tensor([0.9, 1.0, 1.1])
    if not torch.isin(multiplier, allowed).all():
        raise RuntimeError(f'Exp{number} emitted unsupported multiplier')
    if not torch.isin(requested, allowed).all():
        raise RuntimeError(
            f'Exp{number} emitted unsupported requested multiplier')
    if number == 49:
        if not eligible.all():
            raise RuntimeError('Exp49 did not cover every training sample')
    elif number == 50:
        if not torch.equal(eligible, applied):
            raise RuntimeError(
                'Exp50 virtual-focal eligibility differs from MixUp success')
        if not torch.all(multiplier[~applied] == 1.0):
            raise RuntimeError('Exp50 changed a non-MixUp sample focal length')

    val_batch = next(iter(val_loader))
    val_targets = val_batch[2]
    if val_targets['mixup_virtual_focal_eligible'].bool().any():
        raise RuntimeError(f'Exp{number} augmented validation samples')
    if not torch.all(
            val_targets['mixup_virtual_focal_multiplier'].float() == 1.0):
        raise RuntimeError(f'Exp{number} changed validation focal length')
    receipt = {
        'batch_size': int(train_batch[0].shape[0]),
        'workers': int(train_loader.num_workers),
        'mixup_applied': int(applied.sum()),
        'virtual_focal_eligible': int(eligible.sum()),
        'requested_multipliers': sorted(set(
            float(value) for value in requested.tolist())),
        'actual_multipliers': sorted(set(
            float(value) for value in multiplier.tolist())),
        'validation_virtual_focal_eligible': int(
            val_targets['mixup_virtual_focal_eligible'].sum()),
    }
    del train_batch, val_batch, train_loader, val_loader
    gc.collect()
    return receipt


def _no_quality_model_step_receipt(number):
    """Run one real BS1 optimizer step for an Exp47-derived candidate."""
    cfg = _preflight_config(number)
    set_random_seed(cfg.get('random_seed', 444))
    train_loader, _ = build_dataloader(cfg['dataset'], workers=4)
    train_loader.dataset.set_epoch(0)
    inputs, calibs, targets, _ = _move_batch_to_cuda(
        next(iter(train_loader)))
    img_sizes = targets.get('model_image_size', targets['img_size'])
    prepared = _prepare_targets(targets, inputs.shape[0])
    model, criterion = build_model(cfg['model'])
    model = model.cuda().train()
    criterion = criterion.cuda().train()
    optimizer = build_optimizer(cfg['optimizer'], model)
    class_parameter = next(
        parameter for name, parameter in model.named_parameters()
        if 'class_embed' in name and parameter.requires_grad)
    before = class_parameter.detach().clone()
    optimizer.zero_grad()
    outputs = model(inputs, calibs, prepared, img_sizes, dn_args=None)
    if 'pred_quality' in outputs:
        raise RuntimeError(f'Exp{number} unexpectedly produced pred_quality')
    losses = criterion(outputs, prepared, mask_dict=None)
    total = _weighted_loss(losses, criterion.weight_dict)
    total.backward()
    optimizer.step()
    if torch.equal(before, class_parameter.detach()):
        raise RuntimeError(
            f'Exp{number} classification head did not update')
    receipt = {
        'model_step_batch_size': int(inputs.shape[0]),
        'model_step_loss': float(total.detach().cpu()),
        'pred_quality_present': False,
        'classification_head_updated': True,
    }
    del optimizer, criterion, model, outputs, losses, total
    del inputs, calibs, targets, prepared, train_loader
    _release()
    return receipt


def _exp48_receipt():
    cfg = _preflight_config(48)
    set_random_seed(cfg.get('random_seed', 444))
    train_loader, _ = build_dataloader(cfg['dataset'], workers=4)
    train_loader.dataset.set_epoch(0)
    inputs, calibs, targets, _ = _move_batch_to_cuda(
        next(iter(train_loader)))
    img_sizes = targets.get('model_image_size', targets['img_size'])
    prepared = _prepare_targets(targets, inputs.shape[0])
    model, criterion = build_model(cfg['model'])
    model = model.cuda().train()
    criterion = criterion.cuda().train()
    optimizer = build_optimizer(cfg['optimizer'], model)
    class_parameter = next(
        parameter for name, parameter in model.named_parameters()
        if 'class_embed' in name and parameter.requires_grad)
    before = class_parameter.detach().clone()
    optimizer.zero_grad()
    outputs = model(inputs, calibs, prepared, img_sizes, dn_args=None)
    if 'pred_quality' in outputs:
        raise RuntimeError('Exp48 unexpectedly produced pred_quality')
    losses = criterion(outputs, prepared, mask_dict=None)
    if 'monitor_iou_classification_target_mean' not in losses:
        raise RuntimeError('Exp48 did not execute all-query IoU classification')
    total = _weighted_loss(losses, criterion.weight_dict)
    total.backward()
    optimizer.step()
    if torch.equal(before, class_parameter.detach()):
        raise RuntimeError('Exp48 classification head did not update')
    receipt = {
        'batch_size': int(inputs.shape[0]),
        'loss': float(total.detach().cpu()),
        'iou_classification_target_mean': float(
            losses['monitor_iou_classification_target_mean'].detach().cpu()),
        'pred_quality_present': False,
        'classification_head_updated': True,
    }
    del optimizer, criterion, model, outputs, losses, total
    del inputs, calibs, targets, prepared, train_loader
    _release()
    return receipt


def _exp51_receipt():
    cfg = _preflight_config(51)
    set_random_seed(cfg.get('random_seed', 444))
    train_loader, _ = build_dataloader(cfg['dataset'], workers=4)
    train_loader.dataset.set_epoch(0)
    inputs, calibs, targets, _ = _move_batch_to_cuda(
        next(iter(train_loader)))
    img_sizes = targets.get('model_image_size', targets['img_size'])
    prepared = _prepare_targets(targets, inputs.shape[0])
    model, criterion = build_model(cfg['model'])
    model = model.cuda().train()
    criterion = criterion.cuda().train()
    optimizer = build_optimizer(cfg['optimizer'], model)
    quality_parameters = [
        parameter for name, parameter in model.named_parameters()
        if 'iou_quality_embed' in name
    ]
    if not quality_parameters:
        raise RuntimeError('Exp51 has no quality-head parameters')

    criterion.set_epoch(120)
    quality_before = [parameter.detach().clone()
                      for parameter in quality_parameters]
    optimizer.zero_grad()
    outputs = model(inputs, calibs, prepared, img_sizes, dn_args=None)
    losses = criterion(outputs, prepared, mask_dict=None)
    total = _weighted_loss(losses, criterion.weight_dict)
    total.backward()
    if any(parameter.grad is not None for parameter in quality_parameters):
        raise RuntimeError('Exp51 quality head received gradient at E120')
    optimizer.step()
    if any(not torch.equal(before, parameter.detach())
           for before, parameter in zip(quality_before, quality_parameters)):
        raise RuntimeError('Exp51 quality head updated at E120')
    e120_quality_loss = float(sum(
        value.detach() * criterion.weight_dict[key]
        for key, value in losses.items()
        if key.startswith('loss_quality') and key in criterion.weight_dict
    ).cpu())
    del outputs, losses, total

    criterion.set_epoch(121)
    optimizer.zero_grad()
    outputs = model(inputs, calibs, prepared, img_sizes, dn_args=None)
    losses = criterion(outputs, prepared, mask_dict=None)
    quality_total = _weighted_loss(
        losses, criterion.weight_dict, prefix='loss_quality')
    quality_total.backward()
    if any(parameter.grad is None for parameter in quality_parameters):
        raise RuntimeError('Exp51 quality head missed gradient at E121')
    if not all(torch.isfinite(parameter.grad).all()
               for parameter in quality_parameters):
        raise RuntimeError('Exp51 quality-head gradient is non-finite')
    shared_gradients = [
        parameter.grad for name, parameter in model.named_parameters()
        if 'det3d_transformer' in name
        and 'iou_quality_embed' not in name
        and parameter.grad is not None
    ]
    if not shared_gradients or not any(
            gradient.detach().abs().max() > 0
            for gradient in shared_gradients):
        raise RuntimeError(
            'Exp51 quality loss did not backpropagate into shared 3D decoder')
    receipt = {
        'batch_size': int(inputs.shape[0]),
        'e120_quality_loss': e120_quality_loss,
        'e120_quality_grad_none': True,
        'e120_quality_parameters_unchanged': True,
        'e121_quality_loss': float(quality_total.detach().cpu()),
        'e121_quality_grad_finite': True,
        'e121_shared_decoder_grad_nonzero': True,
    }
    del optimizer, criterion, model, outputs, losses, quality_total
    del inputs, calibs, targets, prepared, train_loader
    _release()
    return receipt


def _exp52_receipt():
    cfg = _preflight_config(52)
    parameter = torch.nn.Parameter(torch.ones(1, device='cuda'))
    module = torch.nn.Module()
    module.register_parameter('weight', parameter)
    optimizer = build_optimizer(cfg['optimizer'], module)
    scheduler, warmup = build_lr_scheduler(
        cfg['lr_scheduler'], optimizer, last_epoch=-1)
    if not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        raise RuntimeError('Exp52 did not build CosineAnnealingLR')
    if not isinstance(warmup, torch.optim.lr_scheduler._LRScheduler):
        raise RuntimeError('Exp52 did not build a warmup scheduler')
    if warmup.num_epoch != 10:
        raise RuntimeError('Exp52 warmup does not span 10 epochs')
    lrs = {0: float(optimizer.param_groups[0]['lr'])}
    for epoch_index in range(250):
        optimizer.zero_grad()
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        if epoch_index < warmup.num_epoch:
            warmup.step()
        else:
            scheduler.step()
        epoch = epoch_index + 1
        if epoch in (1, 10, 11, 85, 125, 250):
            lrs[epoch] = float(optimizer.param_groups[0]['lr'])
    if lrs[250] != cfg['lr_scheduler']['eta_min']:
        raise RuntimeError('Exp52 cosine LR did not reach eta_min at E250')
    return {
        'scheduler_class': type(scheduler).__name__,
        'warmup_class': type(warmup).__name__,
        'warmup_epochs': warmup.num_epoch,
        'learning_rates': lrs,
    }


def main():
    started = time.time()
    runtime = _runtime_receipt()
    _enable_exp46_execution_contract()
    git_diff = subprocess.check_output(
        ('git', 'diff', '--binary', 'HEAD'), cwd=ROOT)
    receipt = {
        'status': 'running',
        'runtime': runtime,
        'git_commit': subprocess.check_output(
            ('git', 'rev-parse', 'HEAD'), cwd=ROOT, text=True).strip(),
        'tracked_diff_sha256': hashlib.sha256(git_diff).hexdigest(),
        'strict_determinism': torch.are_deterministic_algorithms_enabled(),
        'fill_uninitialized_memory': (
            torch.utils.deterministic.fill_uninitialized_memory),
        'exp48': _exp48_receipt(),
        'exp49': {
            **_data_receipt(49),
            **_no_quality_model_step_receipt(49),
        },
        'exp50': {
            **_data_receipt(50),
            **_no_quality_model_step_receipt(50),
        },
        'exp51': _exp51_receipt(),
        'exp52': {
            **_no_quality_model_step_receipt(52),
            **_exp52_receipt(),
        },
    }
    receipt['status'] = 'passed'
    receipt['seconds'] = time.time() - started
    print(json.dumps(receipt, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
