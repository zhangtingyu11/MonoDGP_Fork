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
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from lib.helpers.config_helper import load_config
from lib.helpers.utils_helper import set_random_seed


EXPECTED = {
    'executable': '/home/zhangtingyu/Project/Mono3D/MonoDGP/.venv-cu129/bin/python',
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
    resolved_config_path = output_dir / 'resolved_config.yaml'
    resolved_config_path.write_text(
        yaml.safe_dump(
            cfg, allow_unicode=True, sort_keys=False),
        encoding='utf-8')
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
        '正式验证前轻量AP诊断间隔：' + str(
            cfg['trainer'].get('early_validation_interval', 0)),
        '早期AP刷新参与best并计算NMS：' + str(bool(
            cfg['trainer'].get('early_validation_updates_best', False))),
        '主要best选择：' + (
            'BEV NMS '
            f"{cfg['trainer'].get('nms_best_selection', {}).get('bev_iou_threshold')} "
            'Car_3d_moderate_R40'
            if cfg['trainer'].get('nms_best_selection', {}).get(
                'enabled', False)
            else '无NMS Car_3d_moderate_R40')
        + '；排序分数=' + str(
            cfg['tester'].get('primary_quality_score', '历史默认')),
        f"验证置信度门槛：{cfg['tester']['threshold']}",
        '跨焦距供体目标最小有效覆盖率：' + str(
            cfg['dataset'].get('mixup_min_object_valid_ratio', '未配置')),
        'MixUp绑定虚拟焦距：' + str(bool(
            cfg['dataset'].get('mixup_virtual_focal', False))),
        'MixUp虚拟焦距倍率：' + ','.join(map(
            str, cfg['dataset'].get(
                'mixup_virtual_focal_multipliers', ()))),
        'best刷新时BEV NMS阈值：' + ','.join(map(
            str, cfg['tester'].get('best_refresh_bev_nms_thresholds', ()))),
        f"命令：{args.command}",
        f"SHA256 {config_path.name}：{_sha256(config_path)}",
        f"SHA256 resolved_config.yaml：{_sha256(resolved_config_path)}",
        f"SHA256 monodgp.py：{_sha256(ROOT_DIR / 'lib/models/monodgp/monodgp.py')}",
        f"SHA256 matcher.py：{_sha256(ROOT_DIR / 'lib/models/monodgp/matcher.py')}",
        f"SHA256 focal_loss.py：{_sha256(ROOT_DIR / 'lib/losses/focal_loss.py')}",
        f"SHA256 bev_nms_helper.py：{_sha256(ROOT_DIR / 'lib/helpers/bev_nms_helper.py')}",
    ]
    quality_files = (
        ROOT_DIR / 'lib/datasets/kitti/kitti_dataset.py',
        ROOT_DIR / 'lib/datasets/kitti/kitti_utils.py',
        ROOT_DIR / 'lib/datasets/kitti/mixup_geometry.py',
        ROOT_DIR / 'lib/helpers/decode_helper.py',
        ROOT_DIR / 'lib/helpers/tester_helper.py',
        ROOT_DIR / 'lib/helpers/trainer_helper.py',
        ROOT_DIR / 'lib/helpers/quality_ranking_monitor.py',
        ROOT_DIR / 'lib/helpers/nms_best_query_monitor.py',
        ROOT_DIR / 'lib/helpers/swanlab_helper.py',
        ROOT_DIR / 'lib/helpers/gradient_monitor.py',
        ROOT_DIR / 'lib/losses/asymmetric_interval_depth_loss.py',
        ROOT_DIR / 'lib/losses/query_quality_ranking_loss.py',
        ROOT_DIR / 'lib/losses/nms_aware_iou_ranking_loss.py',
        ROOT_DIR / 'lib/models/monodgp/iou3d_match_cost.py',
    )
    receipt.extend(
        f"SHA256 {path.name}：{_sha256(path)}"
        for path in quality_files if path.exists())
    deterministic_extension_dir = (
        ROOT_DIR / 'lib/models/monodgp/ops/deterministic')
    deterministic_binaries = tuple(sorted(
        deterministic_extension_dir.glob(
            'MonoDGPDeterministicMSDA*.so')))
    if (cfg['trainer'].get('strict_determinism', False)
            and not deterministic_binaries):
        raise RuntimeError(
            'strict deterministic run requires the repository-local MSDA '
            'extension; build it before writing the formal run manifest')
    deterministic_extension_files = (
        deterministic_extension_dir / 'msda_deterministic_backward.cu',
        *deterministic_binaries,
    )
    receipt.extend(
        f"SHA256 仓库内确定性MSDA {path.name}：{_sha256(path)}"
        for path in deterministic_extension_files if path.exists())
    receipt.extend([
        '严格确定性算法：' + str(bool(
            cfg['trainer'].get('strict_determinism', False))),
        '批量精确三维IoU匹配：' + str(bool(
            cfg['model'].get('use_batched_iou3d_match_cost', False))),
        '仅主AP验证：' + str(bool(
            cfg['trainer'].get('primary_ap_only_validation', False))),
    ])
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
    quality_cfg = cfg['model'].get('iou_quality_head', {})
    iou_classification_cfg = cfg['model'].get(
        'iou_classification', {})
    nms_ranking_cfg = iou_classification_cfg.get('nms_ranking', {})
    nms_best_cfg = cfg['trainer'].get('nms_best_selection', {})
    score_fusions = cfg['tester'].get('quality_score_fusions', ())
    receipt.extend([
        '三维IoU质量头：' + str(bool(
            quality_cfg.get('enabled', False))),
        '全query分类拟合三维IoU：' + str(bool(
            iou_classification_cfg.get('enabled', False))),
        'IoU分类Quality Focal beta：' + str(
            iou_classification_cfg.get('beta', '未配置')),
        '触发NMS候选对排序：' + str(bool(
            nms_ranking_cfg.get('enabled', False))),
        'NMS排序Loss系数：' + str(
            nms_ranking_cfg.get('loss_coef', '未配置')),
        'NMS排序预测框BEV IoU阈值：' + str(
            nms_ranking_cfg.get('bev_iou_threshold', '未配置')),
        'NMS排序最小真实3D IoU差：' + str(
            nms_ranking_cfg.get('min_iou_delta', '未配置')),
        'NMS排序策略：' + str(
            nms_ranking_cfg.get('strategy', 'all_conflicting_pairs')),
        'NMS排序仅最终Decoder层：' + str(bool(
            nms_ranking_cfg.get('final_layer_only', False))),
        '每轮NMS独立选优：' + str(bool(
            nms_best_cfg.get('enabled', False))),
        '每轮NMS独立选优阈值：' + str(
            nms_best_cfg.get('bev_iou_threshold', '未配置')),
        '每轮NMS对照阈值：' + ','.join(map(
            str, nms_best_cfg.get('report_bev_iou_thresholds', ()))),
        '质量Loss权重：' + str(
            quality_cfg.get('loss_coef', '未配置')),
        '质量监督模式：' + str(
            quality_cfg.get('supervision', 'hungarian_positive')),
        '全query点式Loss系数：' + str(
            quality_cfg.get('point_loss_coef', '未配置')),
        '同GT排序Loss系数：' + str(
            quality_cfg.get('rank_loss_coef', '未配置')),
        '排序query对最小IoU差：' + str(
            quality_cfg.get('ranking_iou_gap', '未配置')),
        '低IoU点式监督阈值/权重：' + str(
            quality_cfg.get('low_iou_threshold', '未配置')) + '/'
            + str(quality_cfg.get('low_iou_weight', '未配置')),
        '点式监督达到满权重的IoU：' + str(
            quality_cfg.get('full_weight_iou', '未配置')),
        '质量目标编码：' + str(
            quality_cfg.get('target_encoding', '未配置')),
        '质量头独立初始化种子：' + str(
            quality_cfg.get('init_seed', '未配置')),
        '主排序分数：' + str(
            cfg['tester'].get('primary_quality_score', '历史默认')),
        '预注册排序组合：' + ';'.join(
            f"{item['name']}(alpha={item.get('alpha', 1.0)},"
            f"beta={item.get('beta', 1.0)},"
            f"gamma={item.get('gamma', 1.0)},"
            f"historical_topk={bool(item.get('historical_topk', False))},"
            f"classification_only={bool(item.get('classification_only', False))})"
            for item in score_fusions),
    ])
    (output_dir / 'run_manifest.txt').write_text(
        '\n'.join(receipt) + '\n', encoding='utf-8')
    print('\n'.join(receipt), flush=True)


if __name__ == '__main__':
    main()
