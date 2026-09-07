import os
import re
import warnings
import torch
import torch.nn as nn


_OBSOLETE_MODEL_STATE_PATTERNS = tuple(re.compile(pattern) for pattern in (
    r'^(?:module\.)?region_head\.deconv\.\d+\.(?:weight|bias)$',
    r'^(?:module\.)?depth_predictor\.(?:depth_bin_values|depth_pos_embed\.weight)$',
    r'^(?:module\.)?(?:(?:det2d|det3d)_transformer\.decoder\.)?'
    r'(?:class_embed|bbox_embed|dim_embed_3d|angle_embed|depth_embed)\.'
    r'\d+\.',
))


def _load_model_state_with_obsolete_key_compatibility(model, model_state,
                                                       logger=None):
    expected_keys = set(model.state_dict())
    removed_keys = [
        key for key in model_state
        if key not in expected_keys and any(
            pattern.match(key)
            for pattern in _OBSOLETE_MODEL_STATE_PATTERNS)]
    if removed_keys:
        model_state = {
            key: value for key, value in model_state.items()
            if key not in removed_keys}
        if logger is not None:
            logger.info(
                'Ignored %d obsolete, non-forward model-state entries: %s',
                len(removed_keys), ', '.join(removed_keys))
    # Keep strict loading for every key outside the explicit compatibility
    # list so a real architecture/checkpoint mismatch cannot pass silently.
    model.load_state_dict(model_state, strict=True)


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_checkpoint_state(model=None, optimizer=None, epoch=None,
                         best_result=None, best_epoch=None,
                         lr_scheduler=None, warmup_lr_scheduler=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {
        'epoch': epoch,
        'model_state': model_state,
        'optimizer_state': optim_state,
        'lr_scheduler_state': (
            lr_scheduler.state_dict() if lr_scheduler is not None else None),
        'warmup_lr_scheduler_state': (
            warmup_lr_scheduler.state_dict()
            if warmup_lr_scheduler is not None else None),
        'best_result': best_result,
        'best_epoch': best_epoch,
    }


def save_checkpoint(state, filename):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename, map_location, logger=None,
                    lr_scheduler=None, warmup_lr_scheduler=None):
    if os.path.isfile(filename):
        logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location)
        epoch = checkpoint.get('epoch', -1)
        best_result = checkpoint.get('best_result', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0.0)
        if model is not None and checkpoint['model_state'] is not None:
            _load_model_state_with_obsolete_key_compatibility(
                model, checkpoint['model_state'], logger=logger)
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        warmup_epochs = int(getattr(
            warmup_lr_scheduler, 'num_epoch', 5))
        if lr_scheduler is not None:
            scheduler_state = checkpoint.get('lr_scheduler_state')
            if scheduler_state is not None:
                lr_scheduler.load_state_dict(scheduler_state)
            elif epoch >= 0 and not (
                    warmup_lr_scheduler is not None
                    and epoch <= warmup_epochs):
                # Historical checkpoints predate scheduler serialization.
                # Reconstruct the exact closed-form state after ``epoch``
                # completed epochs.  The next resumed epoch must train with
                # lr(epoch), then advance to lr(epoch + 1), exactly like an
                # uninterrupted run.
                completed_scheduler_steps = (
                    epoch - warmup_epochs
                    if warmup_lr_scheduler is not None else epoch)
                loaded_lrs = [
                    group['lr'] for group in lr_scheduler.optimizer.param_groups]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    lr_scheduler.step(completed_scheduler_steps)
                # The optimizer checkpoint already contains the exact
                # recursively-computed LR, which can differ by a few ulps from
                # the scheduler's closed form.  Keep that exact value while
                # retaining the reconstructed epoch/state for the next step.
                for group, loaded_lr in zip(
                        lr_scheduler.optimizer.param_groups, loaded_lrs):
                    group['lr'] = loaded_lr
                lr_scheduler._last_lr = loaded_lrs
        if warmup_lr_scheduler is not None:
            warmup_state = checkpoint.get('warmup_lr_scheduler_state')
            if warmup_state is not None:
                warmup_lr_scheduler.load_state_dict(warmup_state)
            elif 0 <= epoch <= warmup_epochs:
                loaded_lrs = [
                    group['lr']
                    for group in warmup_lr_scheduler.optimizer.param_groups]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    warmup_lr_scheduler.step(epoch)
                for group, loaded_lr in zip(
                        warmup_lr_scheduler.optimizer.param_groups,
                        loaded_lrs):
                    group['lr'] = loaded_lr
                warmup_lr_scheduler._last_lr = loaded_lrs
        logger.info("==> Done")
    else:
        raise FileNotFoundError

    return epoch, best_result, best_epoch
