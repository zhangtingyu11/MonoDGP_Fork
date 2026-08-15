"""Validate the formal MonoDGP runtime and write its pre-run receipt."""

import argparse
import datetime
import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

import numba
import torch
import torchvision


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from lib.helpers.config_helper import load_config
from lib.helpers.utils_helper import set_random_seed


EXPECTED = {
    'executable': str(ROOT_DIR / '.venv-cu129/bin/python'),
    'python': '3.10.20',
    'torch': '2.8.0+cu129',
    'torchvision': '0.23.0+cu129',
    'cuda': '12.9',
    'cudnn': 91002,
    'numba': '0.66.0',
    'numba_cuda': '0.30.4',
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args):
    return subprocess.check_output(
        ('git', *args), cwd=ROOT_DIR, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--command', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_random_seed(cfg.get('random_seed', 444))
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
        key: (EXPECTED[key], actual[key])
        for key in EXPECTED if actual[key] != EXPECTED[key]
    }
    if mismatches:
        raise RuntimeError(f'formal environment mismatch: {mismatches}')
    if not torch.cuda.is_available():
        raise RuntimeError('formal run has no host-visible CUDA device')

    output_dir = ROOT_DIR / cfg['trainer']['save_path'] / cfg['model_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    tracked_diff = subprocess.check_output(
        ('git', 'diff', '--binary', 'HEAD'), cwd=ROOT_DIR)
    receipt = [
        f"实验：{cfg['trainer']['swanlab']['experiment_name']}",
        f"启动时间：{datetime.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"代码分支：{_git('branch', '--show-current')}",
        f"代码基线提交：{_git('rev-parse', 'HEAD')}",
        f"已跟踪未提交差异SHA256：{hashlib.sha256(tracked_diff).hexdigest()}",
        f"解释器：{actual['executable']}",
        f"Python：{actual['python']}",
        f"PyTorch：{actual['torch']}",
        f"torchvision：{actual['torchvision']}",
        f"CUDA：{actual['cuda']}",
        f"cuDNN：{actual['cudnn']}",
        f"Numba：{actual['numba']}",
        f"Numba-CUDA：{actual['numba_cuda']}",
        f"CUDA_VISIBLE_DEVICES：{os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        f"TF32 matmul（调用set_random_seed后）：{torch.backends.cuda.matmul.allow_tf32}",
        f"TF32 cuDNN（调用set_random_seed后）：{torch.backends.cudnn.allow_tf32}",
        f"随机种子：{cfg.get('random_seed', 444)}",
        f"批大小：{cfg['dataset']['batch_size']}",
        f"验证起始轮次：{cfg['trainer']['validation_start_epoch']}",
        f"best选择：无NMS Car_3d_moderate_R40",
        f"验证置信度门槛：{cfg['tester']['threshold']}",
        'best刷新时BEV NMS阈值：' + ','.join(map(
            str, cfg['tester'].get('best_refresh_bev_nms_thresholds', ()))),
        f"命令：{args.command}",
        f"SHA256 {config_path.name}：{_sha256(config_path)}",
        f"SHA256 monodgp.py：{_sha256(ROOT_DIR / 'lib/models/monodgp/monodgp.py')}",
        f"SHA256 matcher.py：{_sha256(ROOT_DIR / 'lib/models/monodgp/matcher.py')}",
        f"SHA256 focal_loss.py：{_sha256(ROOT_DIR / 'lib/losses/focal_loss.py')}",
        f"SHA256 bev_nms_helper.py：{_sha256(ROOT_DIR / 'lib/helpers/bev_nms_helper.py')}",
    ]
    high_iou_weighting = cfg['model'].get(
        'high_iou_unmatched_negative_weighting', {})
    receipt.extend([
        '高IoU未匹配负分类软减弱：' + str(bool(
            high_iou_weighting.get('enabled', False))),
        '负分类完整权重截止IoU：' + str(
            high_iou_weighting.get('full_weight_below_iou', '未配置')),
        '负分类零权重起始IoU：' + str(
            high_iou_weighting.get('zero_weight_at_iou', '未配置')),
    ])
    (output_dir / 'run_manifest.txt').write_text(
        '\n'.join(receipt) + '\n', encoding='utf-8')
    print('\n'.join(receipt), flush=True)


if __name__ == '__main__':
    main()
