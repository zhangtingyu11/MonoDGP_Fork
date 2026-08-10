#!/usr/bin/env python3
"""Real KITTI multi-batch contract and one-epoch foreach AdamW benchmark."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import threading
import time

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.dataloader_helper import build_dataloader  # noqa: E402
from lib.helpers.model_helper import build_model  # noqa: E402
from lib.helpers.optimizer_helper import AdamW, build_optimizer  # noqa: E402
from lib.helpers.scheduler_helper import build_lr_scheduler  # noqa: E402
from lib.helpers.trainer_helper import CudaBatchPrefetcher, Trainer  # noqa: E402
from lib.helpers.utils_helper import set_random_seed  # noqa: E402


def load_config(foreach):
    with (ROOT / 'configs/monodgp.yaml').open(encoding='utf-8') as handle:
        cfg = yaml.load(handle, Loader=yaml.Loader)
    cfg['dataset']['batch_size'] = 16
    cfg['optimizer']['use_foreach_adamw'] = bool(foreach)
    cfg['trainer']['max_epoch'] = 1
    cfg['trainer']['save_frequency'] = 1000
    cfg['trainer']['save_all'] = False
    cfg['trainer']['save_path'] = '/tmp/monodgp_seq6_no_artifacts/'
    return cfg


def environment_receipt():
    return {
        'python_executable': sys.executable,
        'python': sys.version,
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'gpu': torch.cuda.get_device_name(0),
    }


def state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(str(tuple(value.shape)).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class BoardMemorySampler:
    def __init__(self):
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None

    @staticmethod
    def _sample_once():
        completed = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used',
             '--format=csv,noheader,nounits', '--id=0'],
            check=True, capture_output=True, text=True)
        return int(completed.stdout.strip().splitlines()[0])

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self.samples.append(self._sample_once())
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self.stop_event.wait(0.25)

    def __enter__(self):
        self.samples.append(self._sample_once())
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        self.thread.join()
        self.samples.append(self._sample_once())

    @property
    def peak_mib(self):
        return max(self.samples) if self.samples else None


def build_stack(cfg, logger_name):
    set_random_seed(cfg.get('random_seed', 444))
    train_loader, test_loader = build_dataloader(cfg['dataset'])
    model, loss = build_model(cfg['model'])
    model = model.cuda()
    optimizer = build_optimizer(cfg['optimizer'], model)
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(
        cfg['lr_scheduler'], optimizer, last_epoch=-1)
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    trainer = Trainer(
        cfg=cfg['trainer'], model=model, optimizer=optimizer,
        train_loader=train_loader, test_loader=test_loader,
        lr_scheduler=lr_scheduler,
        warmup_lr_scheduler=warmup_lr_scheduler,
        logger=logger, loss=loss, model_name=logger_name)
    return train_loader, model, loss, optimizer, trainer


def make_shadow_optimizer(named_parameters, cfg):
    shadow_pairs = [
        (name, torch.nn.Parameter(param.detach().clone()))
        for name, param in named_parameters]
    biases = [param for name, param in shadow_pairs if 'bias' in name]
    weights = [param for name, param in shadow_pairs if 'bias' not in name]
    optimizer = AdamW([
        {'params': biases, 'weight_decay': 0},
        {'params': weights, 'weight_decay': cfg['weight_decay']},
    ], lr=cfg['lr'], foreach=True)
    return shadow_pairs, optimizer


def assert_step_exact(named_parameters, shadow_pairs, ordinary, foreach):
    mismatches = []
    for (name, param), (shadow_name, shadow) in zip(
            named_parameters, shadow_pairs):
        if name != shadow_name or not torch.equal(param, shadow):
            mismatches.append(f'parameter:{name}')
        ordinary_state = ordinary.state.get(param, {})
        foreach_state = foreach.state.get(shadow, {})
        if tuple(ordinary_state) != tuple(foreach_state):
            mismatches.append(f'state-keys:{name}')
            continue
        for field, left in ordinary_state.items():
            right = foreach_state[field]
            equal = (torch.equal(left, right) if torch.is_tensor(left)
                     else left == right)
            if not equal:
                mismatches.append(f'{field}:{name}')
    if mismatches:
        raise RuntimeError(
            'ordinary and foreach AdamW diverged: ' + ', '.join(mismatches[:8]))


def verify_real_batches(output_path, batch_count):
    cfg = load_config(False)
    train_loader, model, loss, ordinary, trainer = build_stack(
        cfg, 'seq6-real-contract')
    named_parameters = list(model.named_parameters())
    shadow_pairs, foreach = make_shadow_optimizer(
        named_parameters, cfg['optimizer'])
    batch_source = CudaBatchPrefetcher(
        iter(train_loader), torch.device('cuda'),
        copy_stream=trainer.cuda_batch_copy_stream)
    checked = 0
    losses = []
    try:
        for inputs, calibs, targets, _ in batch_source:
            ordinary.zero_grad()
            foreach.zero_grad()
            img_sizes = targets['img_size']
            targets = trainer.prepare_targets(targets, inputs.shape[0])
            dn_args = None
            if trainer.cfg['use_dn']:
                dn_args = (
                    targets, trainer.cfg['scalar'],
                    trainer.cfg['label_noise_scale'],
                    trainer.cfg['box_noise_scale'],
                    trainer.cfg['num_patterns'])
            outputs = model(
                inputs, calibs, targets, img_sizes, dn_args=dn_args)
            loss_dict = loss(outputs, targets, None)
            weight_dict = loss.weight_dict
            total = sum(
                loss_dict[key] * weight_dict[key]
                for key in loss_dict if key in weight_dict)
            total.backward()
            for (_, param), (_, shadow) in zip(
                    named_parameters, shadow_pairs):
                shadow.grad = param.grad
            ordinary.step()
            foreach.step()
            assert_step_exact(
                named_parameters, shadow_pairs, ordinary, foreach)
            checked += 1
            losses.append(float(total.detach().cpu()))
            if checked >= batch_count:
                break
    finally:
        batch_source.close()
    result = {
        'status': 'passed',
        'real_batches_checked': checked,
        'batch_size': 16,
        'real_images_checked': checked * 16,
        'parameter_and_state_exact_after_every_step': True,
        'losses': losses,
        **environment_receipt(),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def benchmark_arm(arm, output_path):
    foreach = arm == 'foreach'
    cfg = load_config(foreach)
    train_loader, model, _, optimizer, trainer = build_stack(
        cfg, f'seq6-{arm}')
    initial_hash = state_sha256(model)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with BoardMemorySampler() as board:
        start = time.perf_counter()
        summary = trainer.train_one_epoch(0)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
    result = {
        'status': 'completed',
        'arm': arm,
        'use_foreach_adamw': optimizer.use_foreach,
        'batch_size': 16,
        'workers': train_loader.num_workers,
        'images': len(train_loader.dataset),
        'steps': len(train_loader),
        'seconds': seconds,
        'images_per_second': len(train_loader.dataset) / seconds,
        'mean_loss': summary['mean_loss'],
        'initial_model_sha256': initial_hash,
        'final_model_sha256': state_sha256(model),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'board_fb_peak_mib': board.peak_mib,
        **environment_receipt(),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode', choices=('verify-real', 'ordinary', 'foreach'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--batches', type=int, default=12)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.mode == 'verify-real':
        verify_real_batches(args.output, args.batches)
    else:
        benchmark_arm(args.mode, args.output)


if __name__ == '__main__':
    main()
