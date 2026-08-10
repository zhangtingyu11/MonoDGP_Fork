"""Small, optional SwanLab adapter for train/validation observability."""

from __future__ import annotations

import math
import os

import torch


GEOMETRY_INTERVAL_CHINESE_NAMES = {
    'matched_count': 'Hungarian匹配预测总数（全部类别）',
    'unique_matched_car_count': '去重后参与匹配的真实车辆数',
    'eligible_car_count': '参与区间统计的车辆匹配预测总数（训练含11组）',
    'supported_at_gt_count': '真值深度处三维IoU达到阈值的车辆匹配预测数',
    'valid_interval_count': '可计算有效深度区间的车辆匹配预测数',
    'fallback_native_count': '无法形成有效区间并回退原始深度损失的匹配预测数',
    'inside_count': '预测深度位于有效区间内的车辆匹配预测数',
    'outside_count': '预测深度位于有效区间外的车辆匹配预测数',
    'valid_interval_fraction': '可形成有效区间的车辆匹配预测占比',
    'outside_fraction': '有效区间预测中深度落在区间外的占比',
    'mean_iou_at_gt': '车辆匹配预测在真值深度处的平均三维IoU',
    'depth_mae_all_car': '全部车辆匹配预测的深度MAE',
    'depth_mae_supported': '具有有效区间的车辆匹配预测深度MAE',
    'depth_mae_unsupported': '无法形成有效区间的车辆匹配预测深度MAE',
    'depth_mae_inside': '预测深度位于有效区间内的匹配预测MAE',
    'depth_mae_outside': '预测深度位于有效区间外的匹配预测MAE',
    'mean_outside_boundary_distance': '区间外匹配预测距最近区间边界的平均距离',
    'left_width_virtual_p10': '左侧区间宽度P10',
    'left_width_virtual_median': '左侧区间宽度中位数',
    'left_width_virtual_p90': '左侧区间宽度P90',
    'right_width_virtual_p10': '右侧区间宽度P10',
    'right_width_virtual_median': '右侧区间宽度中位数',
    'right_width_virtual_p90': '右侧区间宽度P90',
    'total_width_virtual_p10': '总区间宽度P10',
    'total_width_virtual_median': '总区间宽度中位数',
    'total_width_virtual_p90': '总区间宽度P90',
}

LOSS_CHINESE_NAMES = {
    'loss_ce': '分类损失',
    'loss_bbox': '二维框位置损失',
    'loss_giou': '二维框重叠损失',
    'loss_dim': '三维尺寸损失',
    'loss_angle': '航向角损失',
    'loss_depth': '深度均值损失',
    'loss_center': '投影三维中心损失',
    'loss_depth_map': '密集深度图损失',
    'loss_region': '物体区域损失',
}


def chinese_weighted_losses(raw_losses, weight_dict):
    return {
        LOSS_CHINESE_NAMES[key]: value * float(weight_dict[key])
        for key, value in raw_losses.items()
        if key in LOSS_CHINESE_NAMES and key in weight_dict
    }


def chinese_geometry_metrics(summary):
    return {
        GEOMETRY_INTERVAL_CHINESE_NAMES[key]: value
        for key, value in summary.items()
        if key in GEOMETRY_INTERVAL_CHINESE_NAMES
    }


class ScalarMeanAccumulator:
    """Accumulate scalar tensors on their device and synchronize at finalize."""

    def __init__(self):
        self._sums = {}
        self._counts = {}

    def add(self, values):
        for key, value in values.items():
            if not torch.is_tensor(value) or value.numel() != 1:
                continue
            detached = value.detach().reshape(()).to(dtype=torch.float64)
            if key not in self._sums:
                self._sums[key] = torch.zeros_like(detached)
                self._counts[key] = 0
            self._sums[key].add_(detached)
            self._counts[key] += 1

    def finalize(self):
        return {
            key: float((total / self._counts[key]).cpu())
            for key, total in self._sums.items()
        }


class GeometryIntervalAccumulator:
    """Aggregate the historical feasible-depth interval receipt."""

    _COUNT_KEYS = (
        'matched_count', 'unique_matched_car_count', 'eligible_car_count',
        'supported_at_gt_count', 'valid_interval_count',
        'fallback_native_count', 'inside_count', 'outside_count')
    _VECTOR_METRICS = {
        'depth_mae_all_car': 'absolute_error_virtual',
        'depth_mae_supported': 'supported_absolute_error_virtual',
        'depth_mae_unsupported': 'unsupported_absolute_error_virtual',
        'depth_mae_inside': 'inside_absolute_error_virtual',
        'depth_mae_outside': 'outside_absolute_error_virtual',
        'mean_outside_boundary_distance': (
            'outside_boundary_distance_virtual'),
    }

    def __init__(self):
        # Keep the small detached receipts on their producing device and
        # synchronize only once at epoch finalization.  Per-batch .cpu() calls
        # otherwise serialize the training stream hundreds of times per epoch.
        self.counts = {key: None for key in self._COUNT_KEYS}
        self.iou_sums = []
        self.iou_count = 0
        self.metric_sums = {
            key: [] for key in self._VECTOR_METRICS}
        self.metric_counts = {key: 0 for key in self._VECTOR_METRICS}
        self.widths = {'left_width_virtual': [], 'right_width_virtual': []}

    def add(self, receipt):
        if not receipt:
            return
        for key in self._COUNT_KEYS:
            value = receipt[key].detach()
            self.counts[key] = (
                value if self.counts[key] is None
                else self.counts[key] + value)
        iou = receipt['iou_at_gt'].detach().float().reshape(-1)
        if iou.numel():
            self.iou_sums.append(iou.sum())
            self.iou_count += iou.numel()
        for metric, field in self._VECTOR_METRICS.items():
            values = receipt[field].detach().float().reshape(-1)
            if values.numel():
                self.metric_sums[metric].append(values.sum())
                self.metric_counts[metric] += values.numel()
        for field in self.widths:
            values = receipt[field].detach().float().reshape(-1)
            if values.numel():
                self.widths[field].append(values)

    def finalize(self):
        counts = {
            key: (float(value.cpu()) if value is not None else 0.0)
            for key, value in self.counts.items()
        }
        eligible = counts['eligible_car_count']
        valid = counts['valid_interval_count']
        result = dict(counts)
        if self.iou_sums:
            iou_sums = torch.stack(tuple(self.iou_sums)).cpu().tolist()
            iou_sum = sum(iou_sums)
        else:
            iou_sum = 0.0
        result.update({
            'valid_interval_fraction': valid / eligible if eligible else 0.0,
            'outside_fraction': (
                counts['outside_count'] / valid if valid else 0.0),
            'mean_iou_at_gt': (
                iou_sum / self.iou_count if self.iou_count else 0.0),
        })
        for key, chunk_sums in self.metric_sums.items():
            if chunk_sums:
                host_sums = torch.stack(tuple(chunk_sums)).cpu().tolist()
                result[key] = (
                    sum(host_sums) / self.metric_counts[key])
            else:
                result[key] = 0.0
        left = (torch.cat(self.widths['left_width_virtual']).cpu()
                if self.widths['left_width_virtual'] else torch.empty(0))
        right = (torch.cat(self.widths['right_width_virtual']).cpu()
                 if self.widths['right_width_virtual'] else torch.empty(0))
        for name, values in (
                ('left_width_virtual', left),
                ('right_width_virtual', right),
                ('total_width_virtual', left + right)):
            quantiles = (torch.quantile(
                values, values.new_tensor((.1, .5, .9))).tolist()
                if values.numel() else (0.0, 0.0, 0.0))
            result[f'{name}_p10'] = float(quantiles[0])
            result[f'{name}_median'] = float(quantiles[1])
            result[f'{name}_p90'] = float(quantiles[2])
        return result


class SwanLabTracker:
    """Initialize and write SwanLab metrics without making it a hard dependency."""

    def __init__(self, cfg, run_config, output_dir, logger, model_name):
        self.logger = logger
        self.enabled = bool(cfg.get('enabled', False))
        self.required = bool(cfg.get('required', False))
        self.run = None
        self._swanlab = None
        if not self.enabled:
            return

        try:
            import swanlab

            self._swanlab = swanlab
            init_kwargs = {
                'project': str(cfg.get('project', 'MonoDGP')),
                'experiment_name': str(
                    cfg.get('experiment_name', model_name)),
                'description': cfg.get('description'),
                'group': cfg.get('group'),
                'tags': list(cfg.get('tags', ())),
                'config': run_config,
                'logdir': os.path.join(output_dir, 'swanlog'),
                'mode': str(cfg.get('mode', 'online')),
                'id': cfg.get('id'),
                'resume': cfg.get('resume'),
            }
            workspace = cfg.get('workspace')
            if workspace:
                init_kwargs['workspace'] = str(workspace)
            init_kwargs = {
                key: value for key, value in init_kwargs.items()
                if value is not None
            }
            self.run = swanlab.init(**init_kwargs)
            self.logger.info(
                'SWANLAB_INITIALIZED project=%s experiment=%s mode=%s id=%s',
                init_kwargs['project'], init_kwargs['experiment_name'],
                init_kwargs['mode'], getattr(self.run, 'id', None))
        except Exception as error:
            self.enabled = False
            if self.run is not None and self._swanlab is not None:
                try:
                    self._swanlab.finish()
                except Exception:
                    pass
                self.run = None
            if self.required:
                raise RuntimeError(
                    'SwanLab is required but initialization failed') from error
            self.logger.exception(
                'SwanLab initialization failed; continuing without tracking: %s',
                error)

    @property
    def active(self):
        return self.run is not None

    def log(self, values, step=None):
        if not self.active:
            return
        payload = {}
        for key, value in values.items():
            if torch.is_tensor(value):
                value = value.detach().reshape(()).cpu().item()
            if isinstance(value, bool):
                value = int(value)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                payload[str(key)] = value
        if not payload:
            return
        self.run.log(payload, step=step)

    def finish(self):
        if self.run is None:
            return
        self._swanlab.finish()
        self.logger.info('SWANLAB_FINISHED id=%s', getattr(self.run, 'id', None))
        self.run = None
