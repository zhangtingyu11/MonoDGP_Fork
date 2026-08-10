"""One full B16 train epoch plus monitored validation timing receipt."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

import torch
import torchvision
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.dataloader_helper import build_dataloader  # noqa: E402
from lib.helpers.model_helper import build_model  # noqa: E402
from lib.helpers.optimizer_helper import build_optimizer  # noqa: E402
from lib.helpers.scheduler_helper import build_lr_scheduler  # noqa: E402
from lib.helpers.tester_helper import Tester  # noqa: E402
from lib.helpers.trainer_helper import Trainer  # noqa: E402
from lib.helpers.utils_helper import set_random_seed  # noqa: E402


EXPECTED = {
    'python': '3.10.20',
    'torch': '2.8.0+cu129',
    'torchvision': '0.23.0+cu129',
    'cuda': '12.9',
    'cudnn': 91002,
    'numba': '0.66.0',
    'numba_cuda': '0.30.4',
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def environment_receipt(command):
    import numba
    import numba_cuda

    receipt = {
        'python_executable': sys.executable,
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'numba': numba.__version__,
        'numba_cuda': numba_cuda.__version__,
        'git_commit': subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip(),
        'git_status': subprocess.run(
            ['git', 'status', '--short'], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.splitlines(),
        'gpu': torch.cuda.get_device_name(0),
        'cuda_device_count': torch.cuda.device_count(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'command': command,
    }
    actual = {key: receipt[key] for key in EXPECTED}
    if actual != EXPECTED:
        raise RuntimeError(
            f'approved environment mismatch: expected={EXPECTED}, '
            f'actual={actual}')
    return receipt


def model_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    command = ' '.join(sys.argv)

    with (ROOT / 'configs/monodgp.yaml').open(encoding='utf-8') as handle:
        cfg = yaml.load(handle, Loader=yaml.Loader)
    cfg['dataset']['batch_size'] = 16
    cfg['trainer']['max_epoch'] = 1
    cfg['trainer']['save_frequency'] = 1000
    cfg['trainer']['save_all'] = False
    cfg['trainer']['save_path'] = '/tmp/monodgp_geometry_monitor_b16/'
    cfg['trainer']['swanlab']['enabled'] = False
    cfg['tester']['export_predictions'] = False

    set_random_seed(cfg.get('random_seed', 444))
    environment = environment_receipt(command)
    train_loader, val_loader = build_dataloader(cfg['dataset'])
    model, criterion = build_model(cfg['model'])
    model = model.cuda()
    optimizer = build_optimizer(cfg['optimizer'], model)
    scheduler, warmup_scheduler = build_lr_scheduler(
        cfg['lr_scheduler'], optimizer, last_epoch=-1)
    logger = logging.getLogger('geometry-monitor-full-b16')
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    trainer = Trainer(
        cfg=cfg['trainer'], model=model, optimizer=optimizer,
        train_loader=train_loader, test_loader=val_loader,
        lr_scheduler=scheduler, warmup_lr_scheduler=warmup_scheduler,
        logger=logger, loss=criterion,
        model_name='geometry-monitor-full-b16', tracker=None)
    initial_hash = model_sha256(model)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_started = time.perf_counter()
    train_summary = trainer.train_one_epoch(0)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_started
    train_peak_allocated = torch.cuda.max_memory_allocated()
    train_peak_reserved = torch.cuda.max_memory_reserved()

    tester = Tester(
        cfg=cfg['tester'], model=model, dataloader=val_loader,
        logger=logger, train_cfg=cfg['trainer'],
        model_name='geometry-monitor-full-b16', criterion=criterion,
        tracker=None)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    validation_started = time.perf_counter()
    results = tester.inference()
    torch.cuda.synchronize()
    validation_inference_seconds = time.perf_counter() - validation_started
    validation_peak_allocated = torch.cuda.max_memory_allocated()
    validation_peak_reserved = torch.cuda.max_memory_reserved()
    ap_started = time.perf_counter()
    evaluation = tester.evaluate(results, return_metrics=True)
    torch.cuda.synchronize()
    ap_seconds = time.perf_counter() - ap_started

    result = {
        'status': 'completed',
        'experiment': 'current feasible-interval monitoring full B16 timing',
        'design': {
            'batch_size': 16,
            'workers': train_loader.num_workers,
            'train_images': len(train_loader.dataset),
            'train_steps': len(train_loader),
            'validation_images': len(val_loader.dataset),
            'validation_steps': len(val_loader),
            'swanlab_enabled': False,
            'checkpoint_saved': False,
            'prediction_exported': False,
            'geometry_monitoring_enabled': (
                criterion.geometry_interval_monitoring_enabled),
            'training_query_groups': criterion.group_num,
            'interval_bisection_steps': 22,
        },
        'training': {
            'seconds': train_seconds,
            'images_per_second': len(train_loader.dataset) / train_seconds,
            'mean_loss': train_summary['mean_loss'],
            'geometry_interval': train_summary['geometry_interval'],
            'peak_allocated_bytes': train_peak_allocated,
            'peak_reserved_bytes': train_peak_reserved,
            'initial_model_sha256': initial_hash,
            'final_model_sha256': model_sha256(model),
        },
        'validation': {
            'inference_and_monitoring_seconds': validation_inference_seconds,
            'images_per_second': (
                len(val_loader.dataset) / validation_inference_seconds),
            'gpu_ap_seconds': ap_seconds,
            'total_seconds': validation_inference_seconds + ap_seconds,
            'selection_score': evaluation['selection_score'],
            'metrics': evaluation['metrics'],
            'mean_losses': tester.last_loss_summary,
            'geometry_interval': tester.last_geometry_interval_summary,
            'peak_allocated_bytes': validation_peak_allocated,
            'peak_reserved_bytes': validation_peak_reserved,
        },
        'reference': {
            'accepted_group_fix_training_epoch_seconds': [
                100.94503415701911, 100.74045528285205],
            'accepted_prefetch_validation_inference_mean_seconds': (
                38.28806535399053),
            'reference_scope': (
                'References predate SwanLab loss and feasible-interval '
                'monitoring, so differences include monitoring overhead.'),
        },
        'artifacts': {
            'config_sha256': file_sha256(ROOT / 'configs/monodgp.yaml'),
            'script_sha256': file_sha256(Path(__file__)),
        },
        **environment,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'training_seconds': train_seconds,
        'validation_inference_and_monitoring_seconds': (
            validation_inference_seconds),
        'gpu_ap_seconds': ap_seconds,
        'validation_total_seconds': validation_inference_seconds + ap_seconds,
        'output': str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
